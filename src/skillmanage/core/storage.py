"""Checkpoint save/load for SkillBank and PatternBuffers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .models import ActiveSkill, ArchivedSkill, ForgottenSkill, PatternBuffer
from .skill_bank import SkillBank

logger = logging.getLogger(__name__)


def save_checkpoint(
    skill_bank: SkillBank,
    pattern_buffers: Dict[str, PatternBuffer],
    path: str,
    current_round: int = 0,
) -> None:
    """Save skill bank and pattern buffers to disk.

    Args:
        skill_bank: SkillBank to save.
        pattern_buffers: PatternBuffers keyed by task_type.
        path: Directory to save files.
        current_round: Current round number (for metadata).
    """
    base = Path(path) / f"round_{current_round:05d}"
    base.mkdir(parents=True, exist_ok=True)

    # Active skills
    active_data = {
        sid: s.to_dict() for sid, s in skill_bank.active.items()
    }
    _write_json(base / "active.json", active_data)

    # Active embeddings
    if skill_bank._active_embeddings:
        ids = list(skill_bank._active_embeddings.keys())
        embs = np.array([skill_bank._active_embeddings[sid] for sid in ids])
        np.savez(base / "active_embeddings.npz", ids=ids, embeddings=embs)

    # Archive skills
    archive_data = {
        sid: s.to_dict() for sid, s in skill_bank.archive.items()
    }
    _write_json(base / "archive.json", archive_data)

    # Archive embeddings
    if skill_bank._archive_embeddings:
        ids = list(skill_bank._archive_embeddings.keys())
        embs = np.array([skill_bank._archive_embeddings[sid] for sid in ids])
        np.savez(base / "archive_embeddings.npz", ids=ids, embeddings=embs)

    # Forgotten skills
    forgotten_data = {
        sid: s.to_dict() for sid, s in skill_bank.forgotten.items()
    }
    _write_json(base / "forgotten.json", forgotten_data)

    # Forgotten embeddings
    if skill_bank._forgotten_embeddings:
        ids = list(skill_bank._forgotten_embeddings.keys())
        embs = np.array([skill_bank._forgotten_embeddings[sid] for sid in ids])
        np.savez(base / "forgotten_embeddings.npz", ids=ids, embeddings=embs)

    # Pattern buffers
    pb_data = {k: v.to_dict() for k, v in pattern_buffers.items()}
    _write_json(base / "pattern_buffers.json", pb_data)

    # Metadata
    meta = {
        "current_round": current_round,
        "stats": skill_bank.stats(),
    }
    _write_json(base / "checkpoint_meta.json", meta)

    logger.info(
        "Saved checkpoint at round %d to %s (active=%d, archive=%d, forgotten=%d)",
        current_round, path,
        len(skill_bank.active), len(skill_bank.archive), len(skill_bank.forgotten),
    )


def load_checkpoint(
    path: str, embedding_dim: int = 384
) -> tuple[SkillBank, Dict[str, PatternBuffer], int]:
    """Load skill bank and pattern buffers from disk.

    Args:
        path: Directory to load from.
        embedding_dim: Embedding dimension.

    Returns:
        Tuple of (SkillBank, pattern_buffers, current_round).
    """
    base = Path(path)
    skill_bank = SkillBank(embedding_dim=embedding_dim)

    # Active
    active_data = _read_json(base / "active.json")
    for sid, d in active_data.items():
        skill_bank.active[sid] = ActiveSkill.from_dict(d)

    active_emb_path = base / "active_embeddings.npz"
    if active_emb_path.exists():
        loaded = np.load(active_emb_path, allow_pickle=True)  # noqa: S301 trust boundary: local checkpoints only
        for sid, emb in zip(loaded["ids"], loaded["embeddings"]):
            skill_bank._active_embeddings[str(sid)] = emb

    # Archive
    archive_data = _read_json(base / "archive.json")
    for sid, d in archive_data.items():
        skill_bank.archive[sid] = ArchivedSkill.from_dict(d)

    archive_emb_path = base / "archive_embeddings.npz"
    if archive_emb_path.exists():
        loaded = np.load(archive_emb_path, allow_pickle=True)  # noqa: S301
        for sid, emb in zip(loaded["ids"], loaded["embeddings"]):
            skill_bank._archive_embeddings[str(sid)] = emb

    # Forgotten
    forgotten_data = _read_json(base / "forgotten.json")
    for sid, d in forgotten_data.items():
        skill_bank.forgotten[sid] = ForgottenSkill.from_dict(d)

    forgotten_emb_path = base / "forgotten_embeddings.npz"
    if forgotten_emb_path.exists():
        loaded = np.load(forgotten_emb_path, allow_pickle=True)  # noqa: S301
        for sid, emb in zip(loaded["ids"], loaded["embeddings"]):
            skill_bank._forgotten_embeddings[str(sid)] = emb

    # Pattern buffers
    pb_data = _read_json(base / "pattern_buffers.json")
    pattern_buffers = {k: PatternBuffer.from_dict(v) for k, v in pb_data.items()}

    # Metadata
    meta = _read_json(base / "checkpoint_meta.json")
    current_round = meta.get("current_round", 0)

    logger.info(
        "Loaded checkpoint from %s at round %d (active=%d, archive=%d, forgotten=%d)",
        path, current_round,
        len(skill_bank.active), len(skill_bank.archive), len(skill_bank.forgotten),
    )
    return skill_bank, pattern_buffers, current_round


def _write_json(path: Path, data: dict) -> None:
    """Write dict to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict:
    """Read JSON file, return empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
