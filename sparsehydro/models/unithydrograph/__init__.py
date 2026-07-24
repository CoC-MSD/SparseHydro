"""sparsehydro.models.unithydrograph — unit hydrograph models.

This subpackage provides native :class:`~sparsehydro.models.IUnitHydroComponent`
implementations plus sequential event-by-event fitting:

- :class:`~sparsehydro.models.unithydrograph.GammaUH` — Gamma-function UH
- :class:`~sparsehydro.models.unithydrograph.NashUH` — Nash cascade UH
- :class:`~sparsehydro.models.unithydrograph.TriangleUH` — Triangular UH
- :class:`~sparsehydro.models.unithydrograph.RectangleUH` — Rectangular pulse UH
- :class:`~sparsehydro.models.unithydrograph.DecayUH` — Exponential-decay UH
- :class:`~sparsehydro.models.unithydrograph.GammaDelayUH` — Delayed Gamma UH
- :class:`~sparsehydro.models.unithydrograph.PeakTailUH` — Blended peak+tail UH
- :class:`~sparsehydro.models.unithydrograph.SequentialFitter` — sequential event fitting
- :class:`~sparsehydro.models.unithydrograph.SequentialFitSummary` — fitting results

Quick start::

    from sparsehydro.models.unithydrograph import GammaUH, SequentialFitter
    from sparsehydro.events import detect_events

    events, filter_result = detect_events(rain_stormflow_df)
    fitter = SequentialFitter(lambda: GammaUH(), rain_stormflow_df, events)
    summary = fitter.fit(verbose=True)
    print(summary.metrics_summary())
"""

from .models import (
    GammaUH,
    NashUH,
    TriangleUH,
    RectangleUH,
    DecayUH,
    GammaDelayUH,
)
from .peak_tail import PeakTailUH
from .sequential import SequentialFitter, SequentialFitSummary, GlobalSequentialFitter

__all__ = [
    "GammaUH",
    "NashUH",
    "TriangleUH",
    "RectangleUH",
    "DecayUH",
    "GammaDelayUH",
    "PeakTailUH",
    "SequentialFitter",
    "SequentialFitSummary",
    "GlobalSequentialFitter",
]
