"""Experiment tracker: logs task results and skill lifecycle events to JSONL files.

Produces two files:
  - task_log.jsonl:  one line per task (round, task_id, success, skills used, etc.)
  - lifecycle_log.jsonl: one line per lifecycle event (skill created/archived/forgotten/repaired/merged/etc.)

These are independent of checkpoint files and can be analyzed separately.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracks task results and skill lifecycle events to JSONL files.

    Args:
        output_dir: Directory to write log files.
    """

    def __init__(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._task_log_path = os.path.join(output_dir, "task_log.jsonl")
        self._lifecycle_log_path = os.path.join(output_dir, "lifecycle_log.jsonl")

        # Ensure log files exist (append mode — safe for restart)
        for path in [self._task_log_path, self._lifecycle_log_path]:
            with open(path, "a") as f:
                pass

    # ------------------------------------------------------------------
    # Task logging
    # ------------------------------------------------------------------

    def log_task(
        self,
        round_num: int,
        task_id: str,
        task_type: str,
        success: bool,
        reward: float,
        ground_truth: str,
        predicted: str,
        used_skill_ids: List[str],
        used_skill_names: List[str],
        num_active: int,
        num_archive: int,
        num_forgotten: int,
        active_tokens: int,
        phase: str = "train",
        repaired: bool = False,
        repair_retries: int = 0,
        question: str = "",
        **extra: Any,
    ) -> None:
        """Log one task result."""
        record = {
            "round": round_num,
            "phase": phase,
            "task_id": task_id,
            "task_type": task_type,
            "question": question,
            "success": success,
            "reward": reward,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "used_skill_ids": used_skill_ids,
            "used_skill_names": used_skill_names,
            "num_active": num_active,
            "num_archive": num_archive,
            "num_forgotten": num_forgotten,
            "active_tokens": active_tokens,
            "repaired": repaired,
            "repair_retries": repair_retries,
        }
        record.update(extra)
        self._append(self._task_log_path, record)

    # ------------------------------------------------------------------
    # Lifecycle event logging
    # ------------------------------------------------------------------

    def log_skill_created(
        self, round_num: int, skill_id: str, skill_name: str,
        source: str, task_type: str, confidence: float, token_cost: int,
    ) -> None:
        """Skill entered Active."""
        self._log_lifecycle(round_num, "created", skill_id, skill_name, {
            "source": source, "task_type": task_type,
            "confidence": confidence, "token_cost": token_cost,
        })

    def log_skill_archived(
        self, round_num: int, skill_id: str, skill_name: str,
        reason: str, importance: float = 0.0,
        call_count: int = 0, success_rate: float = 0.0,
    ) -> None:
        """Skill moved from Active to Archive."""
        self._log_lifecycle(round_num, "archived", skill_id, skill_name, {
            "reason": reason, "importance": importance,
            "call_count": call_count, "success_rate": success_rate,
        })

    def log_skill_forgotten(
        self, round_num: int, skill_id: str, skill_name: str,
    ) -> None:
        """Skill moved from Archive to Forgotten."""
        self._log_lifecycle(round_num, "forgotten", skill_id, skill_name, {})

    def log_skill_recalled(
        self, round_num: int, skill_id: str, skill_name: str,
        success: bool, recall_count: int,
    ) -> None:
        """Archived skill was recalled and used."""
        self._log_lifecycle(round_num, "recalled", skill_id, skill_name, {
            "success": success, "recall_count": recall_count,
        })

    def log_skill_promoted(
        self, round_num: int, skill_id: str, skill_name: str,
    ) -> None:
        """Skill promoted from Archive back to Active."""
        self._log_lifecycle(round_num, "promoted", skill_id, skill_name, {})

    def log_skill_merged(
        self, round_num: int,
        skill_a_id: str, skill_a_name: str,
        skill_b_id: str, skill_b_name: str,
        merged_id: str, merged_name: str,
    ) -> None:
        """Two skills merged into one."""
        self._log_lifecycle(round_num, "merged", merged_id, merged_name, {
            "from_a": {"id": skill_a_id, "name": skill_a_name},
            "from_b": {"id": skill_b_id, "name": skill_b_name},
        })

    def log_skill_distilled(
        self, round_num: int, skill_id: str, skill_name: str,
        old_tokens: int, new_tokens: int,
    ) -> None:
        """Skill was compressed."""
        self._log_lifecycle(round_num, "distilled", skill_id, skill_name, {
            "old_tokens": old_tokens, "new_tokens": new_tokens,
        })

    def log_skill_repaired(
        self, round_num: int, skill_id: str, skill_name: str,
        version: int, reason: str, retry_success: bool,
    ) -> None:
        """Skill was repaired."""
        self._log_lifecycle(round_num, "repaired", skill_id, skill_name, {
            "version": version, "reason": reason, "retry_success": retry_success,
        })

    def log_warning_attached(
        self, round_num: int, skill_id: str, skill_name: str,
        warning_text: str, total_warnings: int,
    ) -> None:
        """Warning was attached to a skill."""
        self._log_lifecycle(round_num, "warning_attached", skill_id, skill_name, {
            "warning": warning_text[:100], "total_warnings": total_warnings,
        })

    def log_warnings_consolidated(
        self, round_num: int, skill_id: str, skill_name: str,
        old_count: int, new_count: int,
    ) -> None:
        """Warnings were consolidated."""
        self._log_lifecycle(round_num, "warnings_consolidated", skill_id, skill_name, {
            "old_count": old_count, "new_count": new_count,
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_lifecycle(
        self, round_num: int, event: str, skill_id: str, skill_name: str,
        details: Dict[str, Any],
    ) -> None:
        record = {
            "round": round_num,
            "event": event,
            "skill_id": skill_id,
            "skill_name": skill_name,
        }
        record.update(details)
        self._append(self._lifecycle_log_path, record)

    def _append(self, path: str, record: dict) -> None:
        import time
        record["_timestamp"] = time.time()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except OSError as e:
            logger.error("Failed to write tracker record to %s: %s", path, e)
