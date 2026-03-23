"""AgentRunner: orchestrates benchmark execution with skill lifecycle."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..acquisition.alignment import PatternBufferManager
from ..acquisition.collector import CollectionDecider
from ..acquisition.failure_learning import FailureLearner, SkillRepairResult, WarningAttachment
from ..acquisition.formalization import Formalizer
from ..acquisition.segmentation import Segmenter
from ..active.manager import ActiveManager
from ..active.meta_updater import MetaUpdater
from ..archive.archive_manager import ArchiveManager
from ..archive.forgotten_manager import ForgottenManager
from ..config import SkillManageConfig
from ..core.embedding import EmbeddingModel
from ..core.models import Skill
from ..core.retrieval import SkillRetriever
from ..core.skill_bank import SkillBank
from ..core.storage import save_checkpoint
from ..llm.base import BaseLLMClient
from .base import (
    Benchmark,
    InteractiveBenchmark,
    InteractionMode,
    TaskInstance,
    TaskResult,
)

logger = logging.getLogger(__name__)


class AgentRunner:
    """Orchestrates running tasks on a benchmark with skill lifecycle management.

    Wires together: retrieval -> agent execution -> meta update -> acquisition -> active management.

    Args:
        benchmark: The benchmark to run on.
        skill_bank: Shared skill bank.
        embedding_model: Embedding model for encoding.
        llm_client: LLM client for agent and lifecycle operations.
        cfg: Full configuration.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        skill_bank: SkillBank,
        embedding_model: EmbeddingModel,
        llm_client: BaseLLMClient,
        cfg: SkillManageConfig,
        tracker=None,
    ) -> None:
        self.benchmark = benchmark
        self.skill_bank = skill_bank
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.cfg = cfg
        self.tracker = tracker  # Optional ExperimentTracker

        # Lifecycle components
        self.retriever = SkillRetriever()
        self.meta_updater = MetaUpdater()
        self.collector = CollectionDecider()
        self.segmenter = Segmenter()
        self.alignment = PatternBufferManager()
        self.forgotten_mgr = ForgottenManager()
        self.formalizer = Formalizer()
        self.failure_learner = FailureLearner()
        self.archive_mgr = ArchiveManager(self.forgotten_mgr)
        self.active_mgr = ActiveManager()

    def run_task(self, task: TaskInstance, current_round: int) -> TaskResult:
        """Run a single task through the full skill-augmented pipeline.

        Flow:
        1. Retrieve skills from Active (fallback to Archive)
        2. Build prompt and execute agent
        3. Check answer
        4. Update meta for used skills
        5. Run Acquisition (success: extract skill / failure: extract warning)
        6. Run Active management (importance, compression, forgetting)
        7. Tick Archive inactive

        Args:
            task: The task to run.
            current_round: Current round number.

        Returns:
            TaskResult with full execution details.
        """
        # 1. Retrieve skills and execute
        archive_hit = None
        if self.benchmark.get_interaction_mode() == InteractionMode.SINGLE_TURN:
            # Single-turn: retrieve by task.instruction, then run
            active_skills, archive_hit = self.retriever.retrieve(
                task.instruction, self.skill_bank, self.embedding_model, self.cfg.retrieval,
            )
            used_skills: List[Skill] = list(active_skills)
            if archive_hit is not None:
                used_skills = [archive_hit.original_skill_full]
            skills_prompt = SkillRetriever.format_skills_for_prompt(used_skills)
            warnings_section = SkillRetriever.format_warnings_for_system(
                used_skills, self.cfg.acquisition.max_warnings
            )
            agent_output = self._run_single_turn(task, skills_prompt, warnings_section)
        else:
            # Multi-step: reset env first to get instruction, then retrieve
            agent_output, used_skills, archive_hit = self._run_multi_step(task)

        # 3. Check answer
        success, reward = self.benchmark.check_answer(task, agent_output)

        # 3.5 Skill repair loop (training only, single-turn with skills)
        if (not success
                and used_skills
                and self.benchmark.get_interaction_mode() == InteractionMode.SINGLE_TURN):
            success, reward, agent_output, used_skills = self._try_repair_and_retry(
                task, agent_output, used_skills, current_round,
            )

        # 4. Extract trajectory
        trajectory = self.benchmark.extract_trajectory(agent_output)

        # For multi-step, get task_type from benchmark (not from frozen TaskInstance)
        task_type = task.task_type
        if not task_type and hasattr(self.benchmark, "current_task_type"):
            task_type = self.benchmark.current_task_type

        result = TaskResult(
            task_id=task.task_id,
            task_type=task_type,
            success=success,
            reward=reward,
            trajectory=trajectory,
            used_skill_ids=[s.skill_id for s in used_skills],
            agent_answer=agent_output,
            ground_truth=task.ground_truth,
            num_steps=len(trajectory),
        )

        # 5. Handle Archive recall result
        if archive_hit is not None:
            if self.tracker:
                archived = self.skill_bank.get_archived_skill(archive_hit.original_skill_id)
                self.tracker.log_skill_recalled(
                    current_round, archive_hit.original_skill_id,
                    archive_hit.original_skill_full.name,
                    success=success,
                    recall_count=archived.recall_count if archived else 0,
                )
            promoted = self.archive_mgr.record_recall_result(
                self.skill_bank, archive_hit.original_skill_id,
                success, current_round, self.embedding_model, self.cfg.archive,
            )
            if promoted:
                logger.info("Archive skill promoted back to Active")
                if self.tracker:
                    self.tracker.log_skill_promoted(
                        current_round, archive_hit.original_skill_id,
                        archive_hit.original_skill_full.name,
                    )

        # 6. Update meta for Active skills used
        active_used_ids = [
            s.skill_id for s in used_skills
            if s.skill_id in self.skill_bank.active
        ]
        self.meta_updater.update_after_task(
            self.skill_bank, active_used_ids, success, reward, current_round,
        )

        # 7. Run Acquisition pipeline
        self._run_acquisition(task, result, used_skills, current_round)

        # 8. Active management (every round)
        active_before = set(self.skill_bank.active.keys())
        archive_before = set(self.skill_bank.archive.keys())

        round_report = self.active_mgr.on_round_end(
            self.skill_bank, current_round, self.llm_client,
            self.embedding_model, self.cfg.active, self.cfg.retrieval,
        )

        # Track skills that moved Active→Archive
        if self.tracker:
            active_after = set(self.skill_bank.active.keys())
            newly_archived = active_before - active_after
            for sid in newly_archived:
                archived = self.skill_bank.get_archived_skill(sid)
                if archived:
                    imp = round_report.importance_scores.get(sid, 0.0)
                    self.tracker.log_skill_archived(
                        current_round, sid, archived.original_skill_full.name,
                        reason="importance" if imp > 0 else "quality_floor",
                        importance=imp,
                        call_count=archived.original_skill_full.token_cost,
                        success_rate=0.0,
                    )

        # 9. Tick Archive inactive
        forgotten_before = set(self.skill_bank.forgotten.keys())

        self.archive_mgr.tick_inactive(
            self.skill_bank, current_round, self.cfg.archive, self.cfg.forgotten,
        )

        # Track skills that moved Archive→Forgotten
        if self.tracker:
            forgotten_after = set(self.skill_bank.forgotten.keys())
            newly_forgotten = forgotten_after - forgotten_before
            for sid in newly_forgotten:
                f = self.skill_bank.forgotten.get(sid)
                if f:
                    self.tracker.log_skill_forgotten(current_round, sid, f.name)

        # 10. Track task result
        if self.tracker:
            predicted = self.benchmark.extract_answer(agent_output) if hasattr(self.benchmark, 'extract_answer') else ""
            stats = self.skill_bank.stats()
            self.tracker.log_task(
                round_num=current_round,
                task_id=task.task_id,
                task_type=task_type,
                success=success,
                reward=reward,
                ground_truth=task.ground_truth,
                predicted=predicted,
                used_skill_ids=[s.skill_id for s in used_skills],
                used_skill_names=[s.name for s in used_skills],
                num_active=stats["active"],
                num_archive=stats["archive"],
                num_forgotten=stats["forgotten"],
                active_tokens=stats["active_tokens"],
            )

        return result

    def run_stream(
        self,
        tasks: List[TaskInstance],
        start_round: int = 0,
        eval_tasks: Optional[List[TaskInstance]] = None,
        eval_interval: int = 500,
    ) -> List[TaskResult]:
        """Run a sequential task stream with periodic evaluation.

        Args:
            tasks: Task stream (train tasks for skill accumulation).
            start_round: Starting round number.
            eval_tasks: Test tasks for periodic evaluation (None = skip).
            eval_interval: Evaluate every N rounds.

        Returns:
            List of TaskResult from the stream (not eval).
        """
        results: List[TaskResult] = []

        for i, task in enumerate(tasks):
            current_round = start_round + i
            result = self.run_task(task, current_round)
            results.append(result)

            # Log progress
            if (i + 1) % 50 == 0:
                recent = results[-50:]
                recent_sr = sum(1 for r in recent if r.success) / len(recent)
                stats = self.skill_bank.stats()
                logger.info(
                    "Round %d: recent_sr=%.2f, active=%d, archive=%d, forgotten=%d, tokens=%d",
                    current_round, recent_sr,
                    stats["active"], stats["archive"], stats["forgotten"],
                    stats["active_tokens"],
                )

            # Periodic evaluation
            if eval_tasks and (i + 1) % eval_interval == 0:
                eval_sr = self.evaluate(eval_tasks)
                logger.info("=== Eval at round %d: SR=%.4f ===", current_round, eval_sr)

            # Checkpoint
            if (i + 1) % self.cfg.checkpoint_interval == 0:
                save_checkpoint(
                    self.skill_bank, self.alignment.buffers,
                    self.cfg.storage_dir, current_round,
                )

        return results

    def evaluate(self, tasks: List[TaskInstance]) -> float:
        """Evaluate on test tasks (frozen skill bank, no updates).

        Args:
            tasks: Test tasks.

        Returns:
            Success rate (0-1).
        """
        correct = 0
        total = 0
        for task in tasks:
            try:
                if self.benchmark.get_interaction_mode() == InteractionMode.SINGLE_TURN:
                    active_skills, archive_hit = self.retriever.retrieve(
                        task.instruction, self.skill_bank, self.embedding_model, self.cfg.retrieval,
                    )
                    used_skills = list(active_skills)
                    if archive_hit:
                        used_skills = [archive_hit.original_skill_full]
                    skills_prompt = SkillRetriever.format_skills_for_prompt(used_skills)
                    warnings_section = SkillRetriever.format_warnings_for_system(
                        used_skills, self.cfg.acquisition.max_warnings
                    )
                    agent_output = self._run_single_turn(task, skills_prompt, warnings_section)
                else:
                    agent_output, _, _ = self._run_multi_step(task)

                success, _ = self.benchmark.check_answer(task, agent_output)
                if success:
                    correct += 1
                total += 1
            except Exception as e:
                logger.error("Evaluate task %s failed: %s", task.task_id, e)
                total += 1

        sr = correct / total if total > 0 else 0.0
        return sr

    # ------------------------------------------------------------------
    # Skill repair + retry
    # ------------------------------------------------------------------

    def _try_repair_and_retry(
        self,
        task: TaskInstance,
        agent_output: str,
        used_skills: List[Skill],
        current_round: int,
    ) -> tuple:
        """Diagnose skill failure and repair+retry if skill is faulty.

        Only runs during training, for single-turn tasks with skills.

        Returns:
            Tuple of (success, reward, agent_output, used_skills) after repair attempts.
        """
        max_retries = self.cfg.acquisition.max_skill_retries
        agent_answer = self.benchmark.extract_answer(agent_output)

        for retry in range(max_retries):
            # Find the first Active skill that was used
            active_skill_to_repair = None
            for s in used_skills:
                if s.skill_id in self.skill_bank.active:
                    active_skill_to_repair = s
                    break

            if active_skill_to_repair is None:
                break

            # Diagnose: model error or skill fault?
            trajectory = self.benchmark.extract_trajectory(agent_output)
            diagnosis = self.failure_learner.diagnose_skill_failure(
                task.instruction, trajectory,
                active_skill_to_repair,
                task.ground_truth, agent_answer,
                self.llm_client,
            )

            if diagnosis is None:
                # Model error, not skill fault — don't repair
                logger.debug("Retry %d: model error, not skill fault. Stopping.", retry + 1)
                break

            # Skill is faulty — repair it
            logger.info(
                "Retry %d/%d: repairing skill '%s' — %s",
                retry + 1, max_retries,
                active_skill_to_repair.name, diagnosis.reason[:60],
            )

            repaired = self.failure_learner.repair_skill(
                active_skill_to_repair,
                task.instruction, diagnosis.reason, task.ground_truth,
                self.llm_client,
            )

            # Update skill in Active
            active_entry = self.skill_bank.get_active_skill(repaired.skill_id)
            if active_entry is None:
                logger.warning("Skill '%s' no longer in Active, aborting repair.", repaired.skill_id)
                break
            active_entry.skill = repaired
            active_entry.version += 1
            new_emb = self.embedding_model.encode_skill(repaired)
            self.skill_bank.update_active_embedding(repaired.skill_id, new_emb)

            if self.tracker:
                self.tracker.log_skill_repaired(
                    current_round, repaired.skill_id, repaired.name,
                    version=active_entry.version,
                    reason=diagnosis.reason[:100],
                    retry_success=False,  # updated below if success
                )

            # Retry with repaired skill
            used_skills = [repaired]
            skills_prompt = SkillRetriever.format_skills_for_prompt(used_skills)
            warnings_section = SkillRetriever.format_warnings_for_system(
                used_skills, self.cfg.acquisition.max_warnings
            )
            agent_output = self._run_single_turn(task, skills_prompt, warnings_section)
            success, reward = self.benchmark.check_answer(task, agent_output)
            agent_answer = self.benchmark.extract_answer(agent_output)

            if success:
                logger.info(
                    "Retry %d: skill '%s' repaired successfully (v%d)",
                    retry + 1, repaired.name,
                    active_entry.version if active_entry else 0,
                )
                return success, reward, agent_output, used_skills

        # All retries exhausted or model error — return last known result
        return False, 0.0, agent_output, used_skills

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------

    def _run_single_turn(self, task: TaskInstance, skills_prompt: str, warnings_section: str = "") -> str:
        """Execute a single-turn task (MATH, BBH)."""
        prompt = self.benchmark.build_prompt(task, skills_prompt)
        # Build system prompt with warnings if benchmark supports it
        if hasattr(self.benchmark, "build_system_prompt_with_warnings") and warnings_section:
            system_prompt = self.benchmark.build_system_prompt_with_warnings(warnings_section)
        else:
            system_prompt = getattr(self.benchmark, "system_prompt", "")
        return self.llm_client.generate(prompt, system_prompt=system_prompt)

    def _run_multi_step(self, task: TaskInstance) -> tuple:
        """Execute a multi-step task (ALFWorld, WebShop).

        Resets env first to get instruction, then retrieves skills.

        Returns:
            Tuple of (agent_output_str, used_skills_list).
        """
        assert isinstance(self.benchmark, InteractiveBenchmark)

        # Reset env to get task instruction
        obs = self.benchmark.reset_env(task)

        # Now retrieve skills using the actual instruction from env
        instruction = getattr(self.benchmark, "current_instruction", "") or obs
        active_skills, archive_hit = self.retriever.retrieve(
            instruction, self.skill_bank, self.embedding_model, self.cfg.retrieval,
        )
        used_skills: List[Skill] = list(active_skills)
        if archive_hit is not None:
            used_skills = [archive_hit.original_skill_full]

        skills_prompt = SkillRetriever.format_skills_for_prompt(used_skills)
        system_prompt = self.benchmark.build_system_prompt(skills_prompt)

        history: List[str] = []
        trajectory_parts = [f"Observation: {obs}"]
        max_steps = getattr(self.benchmark, "max_steps", 30)

        for step_num in range(max_steps):
            step_prompt = self.benchmark.build_step_prompt(task, obs, history)
            action = self.llm_client.generate(step_prompt, system_prompt=system_prompt)
            action = action.strip().split("\n")[0]

            history.append(f"Action: {action}")
            trajectory_parts.append(f"Action: {action}")

            obs, reward, done = self.benchmark.step(action)
            history.append(f"Observation: {obs}")
            trajectory_parts.append(f"Observation: {obs}")

            if done:
                break

        return "\n".join(trajectory_parts), used_skills, archive_hit

    # ------------------------------------------------------------------
    # Acquisition pipeline
    # ------------------------------------------------------------------

    def _run_acquisition(
        self,
        task: TaskInstance,
        result: TaskResult,
        used_skills: List[Skill],
        current_round: int,
    ) -> None:
        """Run the Acquisition pipeline after task completion."""
        decision = self.collector.decide(
            result.success, result.trajectory, used_skills,
            self.llm_client, self.cfg.acquisition,
        )

        if decision.path == "success":
            self._handle_success_acquisition(task, result, decision, current_round)
        elif decision.path == "failure":
            self._handle_failure_acquisition(task, result, current_round)

    def _handle_success_acquisition(
        self, task: TaskInstance, result: TaskResult, decision, current_round: int
    ) -> None:
        """Success path: segment -> alignment -> formalize -> add to Active."""
        # Determine what to segment
        to_segment = decision.full_trajectory or decision.segments_to_collect or []
        if not to_segment:
            return

        # Segment
        segmented = self.segmenter.segment(
            to_segment, task.task_type, self.llm_client,
        )

        # Add to PatternBuffer and check extraction candidates
        candidates = self.alignment.add_record(
            task.task_type, segmented, self.embedding_model, self.cfg.acquisition,
        )

        # Formalize each candidate
        for candidate in candidates:
            confidence = self.alignment.get_confidence(task.task_type, candidate.pattern_id)
            variants = self.alignment.find_variants(
                task.task_type, candidate, self.embedding_model,
            )

            skill = self.formalizer.formalize(
                candidate, task.task_type, variants, confidence,
                self.llm_client, self.skill_bank, self.embedding_model,
                self.forgotten_mgr, current_round, self.cfg.acquisition,
            )

            if skill is not None:
                emb = self.embedding_model.encode_skill(skill)
                self.skill_bank.add_to_active(skill, emb)
                self.alignment.mark_extracted(task.task_type, candidate.pattern_id)

                if self.tracker:
                    self.tracker.log_skill_created(
                        current_round, skill.skill_id, skill.name,
                        skill.source, skill.task_type, skill.confidence, skill.token_cost,
                    )

                self.active_mgr.on_new_skill_added(
                    self.skill_bank, current_round, self.llm_client,
                    self.embedding_model, self.cfg.active, self.cfg.retrieval,
                )

    def _consolidate_warnings(self, active_skill, max_warnings: int) -> None:
        """Consolidate excessive warnings into top-N via LLM."""
        warnings = active_skill.skill.warnings
        prompt = (
            f"The following skill has {len(warnings)} warnings. "
            f"Consolidate them into the {max_warnings} most important, distinct ones.\n\n"
            f"Skill: {active_skill.skill.name}\n"
            f"Warnings:\n" + "\n".join(f"- {w}" for w in warnings) + "\n\n"
            f"Return exactly {max_warnings} consolidated warnings as a JSON list of strings."
        )
        try:
            result = self.llm_client.generate_json(prompt)
            if isinstance(result, list):
                active_skill.skill.warnings = result[:max_warnings]
            elif isinstance(result, dict) and "warnings" in result:
                active_skill.skill.warnings = result["warnings"][:max_warnings]
            logger.info(
                "Consolidated warnings for '%s': %d -> %d",
                active_skill.skill.name, len(warnings), len(active_skill.skill.warnings),
            )
        except (ValueError, KeyError) as e:
            # Fallback: just keep the most recent N
            active_skill.skill.warnings = warnings[-max_warnings:]
            logger.warning("Warning consolidation failed: %s. Keeping last %d.", e, max_warnings)

    def _handle_failure_acquisition(
        self, task: TaskInstance, result: TaskResult, current_round: int
    ) -> None:
        """Failure path: analyze failure -> attach warning or create skill."""
        failure_point = f"Failed after {result.num_steps} steps"

        analysis = self.failure_learner.analyze_failure(
            task.instruction, result.trajectory, failure_point,
            self.skill_bank, self.llm_client, self.embedding_model,
            self.cfg.acquisition,
        )

        if analysis is None:
            return

        if isinstance(analysis, WarningAttachment):
            # Attach warning to existing skill
            active_skill = self.skill_bank.get_active_skill(analysis.skill_id)
            if active_skill and analysis.warning_text not in active_skill.skill.warnings:
                active_skill.skill.warnings.append(analysis.warning_text)
                logger.info(
                    "Attached warning to '%s': %s",
                    active_skill.skill.name, analysis.warning_text[:50],
                )
                if self.tracker:
                    self.tracker.log_warning_attached(
                        current_round, analysis.skill_id, active_skill.skill.name,
                        analysis.warning_text, len(active_skill.skill.warnings),
                    )
                # Consolidate if too many warnings
                max_w = self.cfg.acquisition.max_warnings
                if len(active_skill.skill.warnings) > max_w:
                    old_count = len(active_skill.skill.warnings)
                    self._consolidate_warnings(active_skill, max_w)
                    if self.tracker:
                        self.tracker.log_warnings_consolidated(
                            current_round, analysis.skill_id, active_skill.skill.name,
                            old_count, len(active_skill.skill.warnings),
                        )
        elif isinstance(analysis, Skill):
            # Only create independent failure skill if configured
            if self.cfg.acquisition.create_failure_skill:
                emb = self.embedding_model.encode_skill(analysis)
                self.skill_bank.add_to_active(analysis, emb)
                logger.info("Added failure-derived skill: '%s'", analysis.name)
            else:
                logger.debug("Skipping failure-derived skill (create_failure_skill=False)")
