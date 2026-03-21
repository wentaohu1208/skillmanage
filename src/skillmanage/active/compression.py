"""Compression: Merge similar skills and Distill long skills."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..config import ActiveConfig, RetrievalConfig
from ..core.embedding import EmbeddingModel
from ..core.models import ActiveSkill, Skill, SkillMeta
from ..core.skill_bank import SkillBank
from ..llm.base import BaseLLMClient
from ..llm.prompts import DISTILL_PROMPT, MERGE_PROMPT
from ..utils import count_tokens, generate_skill_id
from ..utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class CompressionReport:
    """Report of compression actions taken."""

    merges: int = 0
    distills: int = 0
    tokens_saved: int = 0
    still_over_budget: bool = False


class Compressor:
    """Orchestrates Merge and Distill to control Active area size.

    Priority order:
    1. Merge (reduces skill count, minimal info loss)
    2. Distill (reduces token cost per skill, some info loss)
    3. Signal for Forgetting if still over budget
    """

    def compress_if_needed(
        self,
        skill_bank: SkillBank,
        embedding_model: EmbeddingModel,
        llm_client: BaseLLMClient,
        active_cfg: ActiveConfig,
        retrieval_cfg: RetrievalConfig,
    ) -> CompressionReport:
        """Run compression cycle if over budget.

        Args:
            skill_bank: The skill bank.
            embedding_model: For computing new embeddings after merge.
            llm_client: For LLM-based merge/distill.
            active_cfg: Active configuration.
            retrieval_cfg: Retrieval configuration (for budget).

        Returns:
            Report of actions taken.
        """
        report = CompressionReport()

        if not skill_bank.is_over_budget(retrieval_cfg):
            return report

        initial_tokens = skill_bank.get_total_active_tokens()
        logger.info(
            "Over budget: %d tokens > %d. Starting compression.",
            initial_tokens, retrieval_cfg.token_budget,
        )

        # Step 1: Merge
        merge_candidates = self._find_merge_candidates(skill_bank, active_cfg)
        for id_a, id_b, sim in merge_candidates:
            if not skill_bank.is_over_budget(retrieval_cfg):
                break
            merged = self._execute_merge(
                id_a, id_b, skill_bank, embedding_model, llm_client
            )
            if merged:
                report.merges += 1

        # Step 2: Distill
        if skill_bank.is_over_budget(retrieval_cfg):
            distill_candidates = self._find_distill_candidates(skill_bank)
            for active_skill in distill_candidates:
                if not skill_bank.is_over_budget(retrieval_cfg):
                    break
                saved = self._execute_distill(
                    active_skill, skill_bank, llm_client
                )
                if saved > 0:
                    report.distills += 1
                    report.tokens_saved += saved

        report.still_over_budget = skill_bank.is_over_budget(retrieval_cfg)
        final_tokens = skill_bank.get_total_active_tokens()
        report.tokens_saved = initial_tokens - final_tokens

        logger.info(
            "Compression done: %d merges, %d distills, saved %d tokens. Over budget: %s",
            report.merges, report.distills, report.tokens_saved, report.still_over_budget,
        )
        return report

    def _find_merge_candidates(
        self, skill_bank: SkillBank, cfg: ActiveConfig
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of skills that should be merged.

        Conditions: sim > tau_merge AND same task_type (or similar structure).

        Returns:
            List of (id_a, id_b, similarity) sorted by similarity descending.
        """
        candidates = []
        ids = skill_bank.active_skill_ids()

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                emb_a = skill_bank.get_active_embedding(id_a)
                emb_b = skill_bank.get_active_embedding(id_b)
                if emb_a is None or emb_b is None:
                    continue

                sim = cosine_similarity(emb_a, emb_b)
                if sim < cfg.merge_threshold:
                    continue

                # Check same task_type
                skill_a = skill_bank.active[id_a].skill
                skill_b = skill_bank.active[id_b].skill
                if skill_a.task_type and skill_b.task_type and skill_a.task_type != skill_b.task_type:
                    continue

                candidates.append((id_a, id_b, sim))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _execute_merge(
        self,
        id_a: str,
        id_b: str,
        skill_bank: SkillBank,
        embedding_model: EmbeddingModel,
        llm_client: BaseLLMClient,
    ) -> Optional[ActiveSkill]:
        """Merge two skills into one.

        Returns:
            The merged ActiveSkill, or None if merge failed.
        """
        as_a = skill_bank.get_active_skill(id_a)
        as_b = skill_bank.get_active_skill(id_b)
        if as_a is None or as_b is None:
            return None

        s_a, s_b = as_a.skill, as_b.skill

        prompt = MERGE_PROMPT.format(
            skill_a_name=s_a.name,
            skill_a_description=s_a.description,
            skill_a_steps="\n".join(s_a.steps),
            skill_a_warnings="\n".join(s_a.warnings) or "none",
            skill_b_name=s_b.name,
            skill_b_description=s_b.description,
            skill_b_steps="\n".join(s_b.steps),
            skill_b_warnings="\n".join(s_b.warnings) or "none",
        )

        try:
            result = llm_client.generate_json(prompt)
        except ValueError as e:
            logger.warning("Merge LLM failed: %s", e)
            return None

        # Create merged skill
        merged_skill = Skill(
            skill_id=generate_skill_id("sk"),
            name=result.get("name", f"{s_a.name}_{s_b.name}"),
            description=result.get("description", s_a.description),
            precondition=result.get("precondition", ""),
            parameters=result.get("parameters", []),
            steps=result.get("steps", s_a.steps),
            warnings=result.get("warnings", []),
            source="merge",
            task_type=s_a.task_type or s_b.task_type,
            confidence=max(s_a.confidence, s_b.confidence),
            token_cost=count_tokens(
                result.get("description", "") + " " + " ".join(result.get("steps", []))
            ),
        )

        # Merge meta (combine counts, weighted average rates)
        total_calls = as_a.meta.call_count + as_b.meta.call_count
        merged_meta = SkillMeta(
            call_count=total_calls,
            success_count=as_a.meta.success_count + as_b.meta.success_count,
            success_rate=(
                (as_a.meta.success_rate * as_a.meta.call_count
                 + as_b.meta.success_rate * as_b.meta.call_count) / max(total_calls, 1)
            ),
            last_used_at=max(as_a.meta.last_used_at, as_b.meta.last_used_at),
            avg_reward=(
                (as_a.meta.avg_reward * as_a.meta.call_count
                 + as_b.meta.avg_reward * as_b.meta.call_count) / max(total_calls, 1)
            ),
        )

        # Remove originals, add merged
        skill_bank.remove_from_active(id_a)
        skill_bank.remove_from_active(id_b)
        merged_emb = embedding_model.encode_skill(merged_skill)
        skill_bank.add_to_active(merged_skill, merged_emb, merged_meta)

        logger.info(
            "Merged '%s' + '%s' -> '%s'",
            s_a.name, s_b.name, merged_skill.name,
        )
        return skill_bank.active[merged_skill.skill_id]

    def _find_distill_candidates(self, skill_bank: SkillBank) -> List[ActiveSkill]:
        """Find skills to distill, sorted by token_cost descending.

        Only skills that haven't been compressed yet.
        """
        candidates = [
            s for s in skill_bank.all_active_skills()
            if not s.compressed
        ]
        candidates.sort(key=lambda s: s.skill.token_cost, reverse=True)
        return candidates

    def _execute_distill(
        self,
        active_skill: ActiveSkill,
        skill_bank: SkillBank,
        llm_client: BaseLLMClient,
    ) -> int:
        """Distill a skill to reduce token cost.

        Returns:
            Tokens saved (0 if distill failed).
        """
        skill = active_skill.skill
        original_cost = skill.token_cost
        target_tokens = max(original_cost // 2, 50)

        prompt = DISTILL_PROMPT.format(
            current_tokens=original_cost,
            skill_name=skill.name,
            skill_description=skill.description,
            skill_steps="\n".join(skill.steps),
            skill_warnings="\n".join(skill.warnings) or "none",
            target_tokens=target_tokens,
        )

        try:
            result = llm_client.generate_json(prompt)
        except ValueError as e:
            logger.warning("Distill LLM failed: %s", e)
            return 0

        new_steps = result.get("steps", skill.steps)
        new_warnings = result.get("warnings", skill.warnings)
        new_token_cost = count_tokens(
            skill.description + " " + " ".join(new_steps)
        )

        # Create new Skill object (avoid mutating shared references)
        updated_skill = Skill(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            precondition=skill.precondition,
            parameters=skill.parameters,
            steps=new_steps,
            warnings=new_warnings,
            source=skill.source,
            task_type=skill.task_type,
            confidence=skill.confidence,
            token_cost=new_token_cost,
        )
        active_skill.skill = updated_skill
        active_skill.compressed = True
        active_skill.version += 1

        saved = original_cost - new_token_cost
        logger.info(
            "Distilled '%s': %d -> %d tokens (saved %d)",
            skill.name, original_cost, skill.token_cost, saved,
        )
        return max(saved, 0)
