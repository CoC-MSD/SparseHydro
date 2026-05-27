"""Private helpers shared across all visualization submodules.

No plotly import — always safe to import regardless of optional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def _resolve_obj(ref: int | str, names: list[str]) -> int:
    """Convert an objective name or index to a zero-based integer index."""
    if isinstance(ref, str):
        if ref not in names:
            raise ValueError(f"Objective {ref!r} not found. Available: {names}")
        return names.index(ref)
    return int(ref)


def _display_col(F: np.ndarray, col: int, minimize_flags: list[bool]) -> np.ndarray:
    """Return column *col* of *F* in display form (un-negated if maximised)."""
    vals = F[:, col]
    return vals if minimize_flags[col] else -vals
