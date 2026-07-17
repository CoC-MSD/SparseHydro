"""Private helpers shared across all visualization submodules.

No plotly import — always safe to import regardless of optional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def _resolve_obj(ref: int | str, names: list[str]) -> int:
    """Convert an objective name or index to a zero-based integer index.

    :param ref: Objective name or zero-based index.
    :type ref: int or str
    :param names: Ordered list of objective names.
    :type names: list[str]
    :returns: Zero-based objective index.
    :rtype: int
    :raises ValueError: If *ref* is a name not present in *names*.
    """
    if isinstance(ref, str):
        if ref not in names:
            raise ValueError(f"Objective {ref!r} not found. Available: {names}")
        return names.index(ref)
    return int(ref)


def _display_col(F: np.ndarray, col: int, minimize_flags: list[bool]) -> np.ndarray:
    """Return column *col* of *F* in display form (un-negated if maximised).

    :param F: Objective matrix in minimisation form, shape ``(n, n_obj)``.
    :type F: numpy.ndarray
    :param col: Index of the objective column to extract.
    :type col: int
    :param minimize_flags: ``True`` for minimised objectives; ``False`` for
        maximised objectives.
    :type minimize_flags: list[bool]
    :returns: The selected column in display form.
    :rtype: numpy.ndarray
    """
    vals = F[:, col]
    return vals if minimize_flags[col] else -vals
