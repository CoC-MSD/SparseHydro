"""Factory helpers for building an additive RTK ensemble.

In SWMM each RTK pathway (fast / medium / slow) has its own independent
initial-abstraction model.  :func:`make_rtk_ensemble` wires up *n_models*
:class:`~sparsehydro.rdii.CombinedHydroModel` instances — each carrying its
own :class:`~sparsehydro.rdii.IAModel` and *n_triangles*
:class:`~sparsehydro.rdii.RTKTriangle` objects — into a single additive
:class:`~sparsehydro.ensemble.EnsembleModel` that can be calibrated end-to-end.

Quick start::

    from sparsehydro.rdii import make_rtk_ensemble

    ensemble = make_rtk_ensemble(
        n_models=3,       # CombinedHydroModel instances (one IAModel each)
        n_triangles=3,    # RTK triangles inside each CombinedHydroModel
        units="imperial",
        area_acres=500.0,
    )
    ensemble.validate()
    ensemble.prepare(df)
    result = ensemble.predict()   # columns: datetime, fast_output, …, rdii_cfs
    ensemble.finalize()
"""

from __future__ import annotations

import math

import numpy as np

from ..ensemble import EnsembleModel
from .combined_model import CombinedHydroModel
from .initial_abstraction import IAModel
from .rtk_triangle import RTKTriangle

_FAST_MEDIUM_SLOW = ["fast", "medium", "slow"]


def default_rtk_params(
    n_models: int,
    n_triangles: int = 1,
    T_min: float = 0.5,
    T_max: float = 120.0,
    R_total: float = 0.10,
    K_min: float = 1.5,
    K_max: float = 3.0,
) -> list[list[tuple[float, float, float]]]:
    """Generate initial ``(R, T, K)`` tuples grouped by model.

    Returns a list of *n_models* sub-lists, each containing *n_triangles*
    ``(R, T, K)`` tuples.  The full set of ``n_models × n_triangles`` triangles
    is distributed across the parameter space before grouping:

    * **T** — log-spaced between *T_min* and *T_max* (decades span is physical).
    * **K** — linearly interpolated from *K_min* (fastest) to *K_max* (slowest).
    * **R** — ``R_total / (n_models × n_triangles)`` for every triangle so the
      calibration starts from an equal baseline.

    Consecutive triangles are assigned to the same model, so model 0 receives
    the fastest T values and model N-1 the slowest.

    :param n_models: Number of CombinedHydroModel instances (>= 1).
    :param n_triangles: RTK triangles per model (>= 1).
    :param T_min: Time-to-peak for the fastest triangle [hr].
    :param T_max: Time-to-peak for the slowest triangle [hr].
    :param R_total: Total runoff fraction divided equally across all triangles.
    :param K_min: Recession ratio for the fastest triangle.
    :param K_max: Recession ratio for the slowest triangle.
    :returns: Nested list ``[[( R, T, K), ...], ...]`` — outer index is model,
        inner index is triangle within that model.
    :rtype: list[list[tuple[float, float, float]]]
    :raises ValueError: If any argument is out of range.
    """
    if n_models < 1:
        raise ValueError(f"n_models must be >= 1; got {n_models!r}")
    if n_triangles < 1:
        raise ValueError(f"n_triangles must be >= 1; got {n_triangles!r}")
    if T_min <= 0.0:
        raise ValueError(f"T_min must be > 0; got {T_min!r}")
    if T_max <= T_min:
        raise ValueError(f"T_max must be > T_min; got T_min={T_min!r}, T_max={T_max!r}")
    if R_total <= 0.0:
        raise ValueError(f"R_total must be > 0; got {R_total!r}")

    n_total = n_models * n_triangles
    R_each = R_total / n_total

    if n_total == 1:
        T_vals = [math.sqrt(T_min * T_max)]
        K_vals = [(K_min + K_max) / 2.0]
    else:
        T_vals = list(np.logspace(math.log10(T_min), math.log10(T_max), n_total))
        K_vals = list(np.linspace(K_min, K_max, n_total))

    flat = [(R_each, float(T), float(K)) for T, K in zip(T_vals, K_vals)]
    return [flat[i * n_triangles:(i + 1) * n_triangles] for i in range(n_models)]


def _make_aliases(n_models: int) -> list[str]:
    if n_models == 3:
        return list(_FAST_MEDIUM_SLOW)
    return [f"rtk_{i}" for i in range(1, n_models + 1)]


def make_rtk_ensemble(
    n_models: int = 3,
    n_triangles: int = 1,
    units: str = "imperial",
    area_acres: float = 100.0,
    ia_defaults: dict | None = None,
    rtk_defaults: list[list[tuple[float, float, float]]] | None = None,
    aliases: list[str] | None = None,
    output_name: str = "rdii_cfs",
) -> EnsembleModel:
    """Build an additive RTK ensemble — one :class:`~sparsehydro.rdii.IAModel`
    per group of RTK triangles.

    Each component is a :class:`~sparsehydro.rdii.CombinedHydroModel` containing
    *n_triangles* :class:`~sparsehydro.rdii.RTKTriangle` objects and one
    :class:`~sparsehydro.rdii.IAModel`.  The *n_models* components are summed
    additively (mixing weights frozen at 1.0) so only the per-component RTK
    and IA parameters are exposed to the optimiser.

    ``area_acres`` is frozen on every child (``calibrate=False``) to prevent
    ``area × R`` identifiability collapse.  To calibrate area post-construction::

        ensemble.set_parameter("fast_area_acres", calibrate=True, ...)

    :param n_models: Number of CombinedHydroModel instances in the ensemble
        (>= 1).  Defaults to 3 (fast / medium / slow).
    :param n_triangles: RTK triangles inside each CombinedHydroModel (>= 1).
        Defaults to 1.
    :param units: Unit system — ``"imperial"`` or ``"metric"``.
    :param area_acres: Drainage area shared by all components [acres].  Frozen
        and excluded from calibration.
    :param ia_defaults: Keyword arguments forwarded to every
        :class:`~sparsehydro.rdii.IAModel` constructor (e.g.
        ``{"snow": True}``).  Each component gets its own independent instance.
    :param rtk_defaults: Nested initial ``(R, T, K)`` values.  Outer list has
        *n_models* entries; each inner list has *n_triangles* ``(R, T, K)``
        tuples.  If omitted, :func:`default_rtk_params` is called.
    :param aliases: Label per component used as a parameter-name prefix.
        Defaults to ``["fast", "medium", "slow"]`` for *n_models*=3, else
        ``["rtk_1", …, "rtk_N"]``.
    :param output_name: Column name for the combined RDII signal in
        ``predict()`` output.  Defaults to ``"rdii_cfs"``.
    :returns: Fully initialised :class:`~sparsehydro.ensemble.EnsembleModel`
        ready for ``validate() → prepare() → predict()``.
    :rtype: :class:`~sparsehydro.ensemble.EnsembleModel`
    :raises ValueError: If argument lengths are inconsistent.

    Example::

        ensemble = make_rtk_ensemble(
            n_models=3,
            n_triangles=3,
            units="imperial",
            area_acres=750.0,
            ia_defaults={"snow": True},
        )
    """
    if n_models < 1:
        raise ValueError(f"n_models must be >= 1; got {n_models!r}")
    if n_triangles < 1:
        raise ValueError(f"n_triangles must be >= 1; got {n_triangles!r}")

    if rtk_defaults is None:
        rtk_defaults = default_rtk_params(n_models, n_triangles)
    if len(rtk_defaults) != n_models:
        raise ValueError(
            f"rtk_defaults length ({len(rtk_defaults)}) must equal n_models ({n_models})"
        )
    for i, group in enumerate(rtk_defaults):
        if len(group) != n_triangles:
            raise ValueError(
                f"rtk_defaults[{i}] length ({len(group)}) must equal "
                f"n_triangles ({n_triangles})"
            )

    if aliases is None:
        aliases = _make_aliases(n_models)
    if len(aliases) != n_models:
        raise ValueError(
            f"aliases length ({len(aliases)}) must equal n_models ({n_models})"
        )

    ia_kwargs: dict = ia_defaults or {}
    children: list[CombinedHydroModel] = []

    for model_rtk in rtk_defaults:
        child = CombinedHydroModel(
            ia_model=IAModel(units=units, **ia_kwargs),
            uh_components=[RTKTriangle(R=R, T=T, K=K) for R, T, K in model_rtk],
            units=units,
        )
        child.initialize()
        # Freeze area: shared physical basin, excluded from optimisation to
        # avoid R×area identifiability collapse.
        child.get_scalar_parameter("area_acres").update(
            value=area_acres, calibrate=False
        )
        children.append(child)

    extractor = lambda df: df["rdii_cfs"].to_numpy()  # noqa: E731
    ensemble = EnsembleModel(
        components=[(child, extractor) for child in children],
        mode="sum",
        aliases=aliases,
        output_name=output_name,
        normalize_weights=False,  # R already partitions rainfall excess
    )
    ensemble.initialize()

    # Freeze mixing weights at 1.0; R in each child handles the fraction
    for i in range(1, n_models + 1):
        ensemble.set_parameter(f"w_{i}", value=1.0, calibrate=False)

    return ensemble
