"""Unique skill ID generation."""

import uuid


def generate_skill_id(prefix: str = "sk") -> str:
    """Generate a unique skill ID.

    Args:
        prefix: Prefix for the ID.

    Returns:
        Unique ID string like 'sk_a1b2c3d4'.
    """
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{short_uuid}"
