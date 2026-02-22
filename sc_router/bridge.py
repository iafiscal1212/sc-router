"""Bridge for optional external classification backends.

Optional: provides hooks for external classification libraries.
Currently no external backends are shipped.
"""

from typing import Dict, Optional

from .catalog import ToolCatalog


def classify_with_sc(
    query: str,
    catalog: ToolCatalog,
) -> Optional[Dict]:
    """Classify using external backend (not currently available)."""
    return None


def hardness_score(query: str, catalog: ToolCatalog) -> Optional[Dict]:
    """Compute hardness score using external backend (not currently available)."""
    return None


def is_available() -> bool:
    """Check if external classification backend is installed."""
    return False
