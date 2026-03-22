"""AgentRunner: orchestrates benchmark execution with skill lifecycle."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..acquisition.alignment import PatternBufferManager
from ..acquisition.collector import CollectionDecider
from ..acquisition.failure_learning import FailureLearner, WarningAttachment
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
    ) -> None:
        self.benchmark = benchmark
        self.skill_bank = skill_bank
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.cfg = cfg

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
            agent_output = self._run_single_turn(task, skills_prompt)
        else:
            # Multi-step: reset env first to get instruction, then retrieve
            agent_output, used_skills = self._run_multi_step(task)

        # 3. Check answer
        success, reward = self.benchmark.check_answer(task, agent_output)

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
            promoted = self.archive_mgr.record_recall_result(
                self.skill_bank, archive_hit.original_skill_id,
                success, current_round, self.embedding_model, self.cfg.archive,
            )
            if promoted:
                logger.info("Archive skill promoted back to Active")

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
        self.active_mgr.on_round_end(
            self.skill_bank, current_round, self.llm_client,
            self.embedding_model, self.cfg.active, self.cfg.retrieval,
        )

        # 9. Tick Archive inactive
        self.archive_mgr.tick_inactive(
            self.skill_bank, current_round, self.cfg.archive, self.cfg.forgotten,
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
                    agent_output = self._run_single_turn(task, skills_prompt)
                else:
                    agent_output, _ = self._run_multi_step(task)

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
    # Execution methods
    # ------------------------------------------------------------------

    def _run_single_turn(self, task: TaskInstance, skills_prompt: str) -> str:
        """Execute a single-turn task (MATH, BBH)."""
        prompt = self.benchmark.build_prompt(task, skills_prompt)
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

        return "\n".join(trajectory_parts), used_skills

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

                self.active_mgr.on_new_skill_added(
                    self.skill_bank, current_round, self.llm_client,
                    self.embedding_model, self.cfg.active, self.cfg.retrieval,
                )

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
        elif isinstance(analysis, Skill):
            # Add new skill from failure analysis
            emb = self.embedding_model.encode_skill(analysis)
            self.skill_bank.add_to_active(analysis, emb)
            logger.info("Added failure-derived skill: '%s'", analysis.name)
