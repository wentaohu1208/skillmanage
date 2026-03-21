"""All configuration as frozen dataclasses."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AcquisitionConfig:
    """Acquisition module hyperparameters."""

    min_records: int = 3
    """M: minimum records in PatternBuffer before extraction."""

    min_confidence: float = 0.5
    """r: minimum pattern confidence (count/total) for extraction."""

    min_failure_steps: int = 2
    """P: minimum execution steps for failure analysis."""

    low_confidence_init: float = 0.5
    """Initial confidence for skills derived from failure analysis."""

    pattern_match_threshold: float = 0.7
    """Semantic similarity threshold for matching segments to existing patterns."""

    cross_category_ratio: float = 0.8
    """If pattern appears in >80% of categories, consider it too generic."""


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval hyperparameters."""

    top_k: int = 3
    """K: number of skills to retrieve per task."""

    similarity_threshold: float = 0.4
    """Minimum cosine similarity for skill retrieval."""

    token_budget: int = 2000
    """Maximum total tokens for Active skill bank."""


@dataclass(frozen=True)
class ActiveConfig:
    """Active area management hyperparameters."""

    merge_threshold: float = 0.8
    """tau_merge: similarity threshold for merging two skills."""

    archive_threshold: float = 0.3
    """theta_archive: importance score below which a skill may be archived."""

    consecutive_rounds: int = 3
    """T1: consecutive rounds below threshold before archiving."""

    weight_recency: float = 0.3
    """w1: weight for Recency in importance score."""

    weight_frequency: float = 0.15
    """w2: weight for Frequency in importance score."""

    weight_quality: float = 0.2
    """w3: weight for Quality in importance score."""

    weight_irreplaceability: float = 0.35
    """w4: weight for Irreplaceability in importance score."""

    recency_decay: float = 0.1
    """lambda: exponential decay rate for Recency."""

    maintain_interval: int = 100
    """N_maintain: rounds between periodic maintenance checks."""


@dataclass(frozen=True)
class ArchiveConfig:
    """Archive hyperparameters."""

    recall_success_threshold: int = 2
    """R: successful recalls needed to promote back to Active."""

    max_inactive_rounds: int = 50
    """Rounds without recall before degrading to Forgotten."""


@dataclass(frozen=True)
class ForgottenConfig:
    """Forgotten area hyperparameters."""

    dedup_threshold: float = 0.8
    """tau_dedup: similarity threshold for deduplication matching."""

    time_decay_rounds: int = 100
    """D: rounds after which a forgotten skill can be re-learned."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str = "/data/hwt/hf_ckpt/Qwen3-Embedding-0.6B"
    dimension: int = 1024


@dataclass(frozen=True)
class LLMConfig:
    """LLM client configuration."""

    provider: str = "openai"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass(frozen=True)
class SkillManageConfig:
    """Top-level configuration."""

    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    active: ActiveConfig = field(default_factory=ActiveConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    forgotten: ForgottenConfig = field(default_factory=ForgottenConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage_dir: str = "outputs/skill_bank"
    checkpoint_interval: int = 50
