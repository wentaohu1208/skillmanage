"""All data model dataclasses with JSON serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Trajectory representation
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A meaningful segment from a trajectory."""

    steps: List[str]
    subgoal: str

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "subgoal": self.subgoal}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Segment:
        return cls(steps=d["steps"], subgoal=d["subgoal"])


@dataclass
class SegmentedTrajectory:
    """A trajectory that has been segmented into meaningful parts."""

    segments: List[Segment]
    task_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SegmentedTrajectory:
        return cls(
            segments=[Segment.from_dict(s) for s in d["segments"]],
            task_type=d.get("task_type", ""),
        )


# ---------------------------------------------------------------------------
# Skill and metadata
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A reusable skill extracted from agent experience."""

    skill_id: str
    name: str
    description: str
    precondition: str = ""
    parameters: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source: str = "experience_distillation"
    task_type: str = ""
    confidence: float = 1.0
    token_cost: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "precondition": self.precondition,
            "parameters": self.parameters,
            "steps": self.steps,
            "warnings": self.warnings,
            "source": self.source,
            "task_type": self.task_type,
            "confidence": self.confidence,
            "token_cost": self.token_cost,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Skill:
        return cls(
            skill_id=d["skill_id"],
            name=d["name"],
            description=d["description"],
            precondition=d.get("precondition", ""),
            parameters=d.get("parameters", []),
            steps=d.get("steps", []),
            warnings=d.get("warnings", []),
            source=d.get("source", "experience_distillation"),
            task_type=d.get("task_type", ""),
            confidence=d.get("confidence", 1.0),
            token_cost=d.get("token_cost", 0),
        )

    def to_prompt_str(self) -> str:
        """Format skill for inclusion in agent prompt."""
        lines = [f"[Skill: {self.name}]"]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.steps:
            lines.append("Steps:")
            for step in self.steps:
                lines.append(f"  {step}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


@dataclass
class SkillMeta:
    """Runtime statistics for an Active skill."""

    call_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    last_used_at: int = 0
    avg_reward: float = 0.0
    low_importance_streak: int = 0

    def update_after_use(self, success: bool, reward: float, current_round: int) -> None:
        """Update meta after skill is used in a task."""
        self.call_count += 1
        if success:
            self.success_count += 1
        self.success_rate = self.success_count / self.call_count
        self.last_used_at = current_round
        # Running average for reward
        alpha = 1.0 / self.call_count
        self.avg_reward = self.avg_reward * (1 - alpha) + reward * alpha

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "last_used_at": self.last_used_at,
            "avg_reward": self.avg_reward,
            "low_importance_streak": self.low_importance_streak,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SkillMeta:
        return cls(
            call_count=d.get("call_count", 0),
            success_count=d.get("success_count", 0),
            success_rate=d.get("success_rate", 0.0),
            last_used_at=d.get("last_used_at", 0),
            avg_reward=d.get("avg_reward", 0.0),
            low_importance_streak=d.get("low_importance_streak", 0),
        )


@dataclass
class ActiveSkill:
    """A skill in the Active area (Skill Bank)."""

    skill: Skill
    meta: SkillMeta = field(default_factory=SkillMeta)
    compressed: bool = False
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill.to_dict(),
            "meta": self.meta.to_dict(),
            "compressed": self.compressed,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ActiveSkill:
        return cls(
            skill=Skill.from_dict(d["skill"]),
            meta=SkillMeta.from_dict(d["meta"]),
            compressed=d.get("compressed", False),
            version=d.get("version", 1),
        )


# ---------------------------------------------------------------------------
# Archive and Forgotten
# ---------------------------------------------------------------------------


@dataclass
class ArchivedSkill:
    """A skill in the Archive area."""

    skill_summary: str
    original_skill_id: str
    original_skill_full: Skill
    archived_at: int = 0
    last_used_at: int = 0
    recall_count: int = 0
    inactive_rounds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_summary": self.skill_summary,
            "original_skill_id": self.original_skill_id,
            "original_skill_full": self.original_skill_full.to_dict(),
            "archived_at": self.archived_at,
            "last_used_at": self.last_used_at,
            "recall_count": self.recall_count,
            "inactive_rounds": self.inactive_rounds,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArchivedSkill:
        return cls(
            skill_summary=d["skill_summary"],
            original_skill_id=d["original_skill_id"],
            original_skill_full=Skill.from_dict(d["original_skill_full"]),
            archived_at=d.get("archived_at", 0),
            last_used_at=d.get("last_used_at", 0),
            recall_count=d.get("recall_count", 0),
            inactive_rounds=d.get("inactive_rounds", 0),
        )


@dataclass
class ForgottenSkill:
    """A skill in the Forgotten area (dedup blacklist)."""

    skill_id: str
    name: str
    summary: str
    forgotten_at: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "summary": self.summary,
            "forgotten_at": self.forgotten_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ForgottenSkill:
        return cls(
            skill_id=d["skill_id"],
            name=d["name"],
            summary=d["summary"],
            forgotten_at=d.get("forgotten_at", 0),
        )


# ---------------------------------------------------------------------------
# Pattern Buffer (for Incremental Alignment)
# ---------------------------------------------------------------------------


@dataclass
class PatternEntry:
    """A single pattern tracked in the PatternBuffer."""

    pattern_id: str
    description: str
    count: int = 1
    extracted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "count": self.count,
            "extracted": self.extracted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PatternEntry:
        return cls(
            pattern_id=d["pattern_id"],
            description=d["description"],
            count=d.get("count", 1),
            extracted=d.get("extracted", False),
        )


@dataclass
class PatternBuffer:
    """Incremental pattern tracking for a task type."""

    task_type: str
    patterns: List[PatternEntry] = field(default_factory=list)
    total_records: int = 0
    extracted_pattern_ids: Set[str] = field(default_factory=set)

    def get_confidence(self, pattern: PatternEntry) -> float:
        """Compute confidence = count / total_records."""
        if self.total_records == 0:
            return 0.0
        return pattern.count / self.total_records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "patterns": [p.to_dict() for p in self.patterns],
            "total_records": self.total_records,
            "extracted_pattern_ids": list(self.extracted_pattern_ids),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PatternBuffer:
        return cls(
            task_type=d["task_type"],
            patterns=[PatternEntry.from_dict(p) for p in d.get("patterns", [])],
            total_records=d.get("total_records", 0),
            extracted_pattern_ids=set(d.get("extracted_pattern_ids", [])),
        )


# ---------------------------------------------------------------------------
# Collection Decision (output of Acquisition collector)
# ---------------------------------------------------------------------------


@dataclass
class CollectionDecision:
    """Result of the collection decision step."""

    path: str  # "success", "failure", "skip"
    segments_to_collect: Optional[List[str]] = None
    full_trajectory: Optional[List[str]] = None
    failure_info: Optional[Dict[str, Any]] = None
    coverage_rate: float = 0.0
