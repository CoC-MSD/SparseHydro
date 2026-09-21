"""Case definitions shared by the golden-fixture generator and its tests.

Kept separate from :mod:`generate` so the test module can enumerate the same
cases without importing generation/CLI machinery.

Every case is a ``(case_id, factory)`` pair where ``factory()`` returns a freshly
built, INITIALIZED model with the case's parameters applied.  Case ids double as
``.npz`` array keys, so they must be stable: renaming one silently drops its pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparsehydro.models.unithydrograph import (
    GammaUH, NashUH, TriangleUH, RectangleUH, DecayUH, GammaDelayUH, PeakTailUH,
)

# Time-step sizes exercised for every kernel case.
DT_CASES: dict[str, float] = {"5min": 5.0 / 60.0, "1hr": 1.0}

# Parameter grids per model.  Each entry is (label, {param: value}).
#
# Coverage intent: defaults, low/mid/high per parameter, degenerate edges, and
# the points where the Step 6 kernel cap becomes active (marked CAP-ACTIVE).
UH_CASES: dict[str, list[tuple[str, dict[str, float]]]] = {
    "GammaUH": [
        ("default", {}),
        ("low", {"tt": 0.01, "tp": 0.01}),
        ("mid", {"tt": 3.0, "tp": 50.0}),
        ("high", {"tt": 5.0, "tp": 200.0}),          # CAP-ACTIVE (1001 -> 864)
        ("extreme", {"tt": 50.0, "tp": 500.0}),      # CAP-ACTIVE (2501 -> 864)
    ],
    "NashUH": [
        ("default", {}),
        ("low", {"n": 0.01, "k": 0.01}),
        ("mid", {"n": 3.0, "k": 10.0}),
        ("high", {"n": 10.0, "k": 20.0}),            # CAP-ACTIVE (1001 -> 864)
        ("overflow", {"n": 100.0, "k": 500.0}),      # CAP-ACTIVE; currently all-NaN
    ],
    "TriangleUH": [
        ("default", {}),
        ("degenerate_tp_ge_tt", {"tt": 10.0, "tp": 20.0}),   # returns zeros(1)
        ("mid", {"tt": 200.0, "tp": 60.0}),
        ("high", {"tt": 1000.0, "tp": 500.0}),       # CAP-ACTIVE (1001 -> 864)
    ],
    "RectangleUH": [
        ("default", {}),
        ("low", {"tr": 1.0}),
        ("mid", {"tr": 200.0}),
        ("high", {"tr": 1000.0}),                    # CAP-ACTIVE (1001 -> 864)
    ],
    "DecayUH": [
        ("default", {}),
        ("zero_alpha", {"alpha": 0.0}),              # returns array([1.0])
        ("mid", {"alpha": 0.9}),
        ("high", {"alpha": 0.999}),
    ],
    "GammaDelayUH": [
        ("default", {}),
        ("no_delay", {"td": 0.0}),
        ("mid", {"tt": 3.0, "tp": 30.0, "td": 20.0}),
        ("high", {"tt": 5.0, "tp": 200.0, "td": 200.0}),
    ],
    "PeakTailUH": [
        ("default", {}),
        ("peak_only", {"w": 0.0}),
        ("tail_only", {"w": 1.0}),
        ("mid", {"w": 0.4, "td": 10.0, "peak_tp": 60.0, "peak_tt": 200.0,
                 "tail_tt": 2.0, "tail_tp": 60.0}),
        ("high", {"w": 0.7, "td": 50.0, "peak_tp": 500.0, "peak_tt": 1000.0,
                  "tail_tt": 5.0, "tail_tp": 500.0}),
    ],
}

_CLASSES = {
    "GammaUH": GammaUH, "NashUH": NashUH, "TriangleUH": TriangleUH,
    "RectangleUH": RectangleUH, "DecayUH": DecayUH,
    "GammaDelayUH": GammaDelayUH, "PeakTailUH": PeakTailUH,
}

# Case ids whose kernels exceed MAX_STEPS today and will therefore change in
# Step 6.  Listed explicitly so the test module can relax its tier for exactly
# these and no others.
CAP_ACTIVE: frozenset[str] = frozenset({
    "GammaUH/high", "GammaUH/extreme",
    "NashUH/high", "NashUH/overflow",
    "TriangleUH/high", "RectangleUH/high",
})


def build(model_name: str, params: dict[str, float]):
    """Return a freshly built, INITIALIZED model with *params* applied.

    :param model_name: Key into the UH class table (e.g. ``"PeakTailUH"``).
    :type model_name: str
    :param params: Flat parameter name -> value overrides applied after
        ``initialize()``.
    :type params: dict[str, float]
    :returns: An initialized model instance.
    :rtype: IUnitHydroComponent
    """
    m = _CLASSES[model_name]()
    m.initialize()
    for k, v in params.items():
        m.apply_flat_parameter(k, float(v))
    return m


def iter_uh_cases():
    """Yield ``(case_id, model_name, params)`` for every UH parameter case.

    :returns: Iterator of ``(str, str, dict)`` triples; ``case_id`` is
        ``"{model_name}/{label}"``.
    :rtype: Iterator[tuple[str, str, dict[str, float]]]
    """
    for model_name, cases in UH_CASES.items():
        for label, params in cases:
            yield f"{model_name}/{label}", model_name, params


def rain_series(n: int, seed: int = 0) -> pd.DataFrame:
    """Return a deterministic synthetic rainfall frame of *n* 5-minute steps.

    Uses a fixed-seed gamma draw with a hard zero floor so the series has
    realistic intermittency (long dry stretches punctuated by bursts) rather
    than uniform drizzle.

    :param n: Number of timesteps.
    :type n: int
    :param seed: PRNG seed.
    :type seed: int
    :returns: DataFrame with ``datetime`` and ``rain`` columns.
    :rtype: pandas.DataFrame
    """
    rng = np.random.default_rng(seed)
    rain = rng.gamma(0.3, 0.05, n)
    rain[rain < 0.01] = 0.0
    return pd.DataFrame(
        {"datetime": pd.date_range("2022-01-01", periods=n, freq="5min"), "rain": rain}
    )


def observed_series(n: int, seed: int = 1) -> np.ndarray:
    """Return a deterministic synthetic observed-flow array with embedded NaNs.

    :param n: Number of timesteps.
    :type n: int
    :param seed: PRNG seed.
    :type seed: int
    :returns: Float array of length *n* with a NaN gap at ``[n//3, n//3 + 5)``.
    :rtype: numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    obs = rng.gamma(2.0, 1.0, n)
    obs[n // 3: n // 3 + 5] = np.nan
    return obs


# Prediction series lengths: 600 straddles the profiled event window (direct
# convolution branch), 5000 straddles the direct/FFT crossover.
PREDICT_LENGTHS: tuple[int, ...] = (600, 5000)
