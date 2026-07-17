"""sparsehydro.models.unithydrograph — unit hydrograph models.

This subpackage provides:

- :class:`~sparsehydro.models.unithydrograph.UnitHydrographAdapter` — the
  adapter base that bridges the existing ``UnitHydrograph`` class to the
  ``IModel`` lifecycle.
- :func:`~sparsehydro.models.unithydrograph.create_uh_model` — factory that
  creates a concrete, registry-compatible subclass for any model registered
  in ``UnitHydrograph._registry``.
- :func:`~sparsehydro.models.unithydrograph.register_all_uh_models` — bulk
  registration helper.
- :class:`~sparsehydro.models.unithydrograph.GammaUH` — Gamma-function UH
- :class:`~sparsehydro.models.unithydrograph.NashUH` — Nash cascade UH
- :class:`~sparsehydro.models.unithydrograph.TriangleUH` — Triangular UH
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

from .adapter import (
    UnitHydrographAdapter,
    create_uh_model,
    register_all_uh_models,
)
from .models import GammaUH, NashUH, TriangleUH
from .sequential import SequentialFitter, SequentialFitSummary

__all__ = [
    "UnitHydrographAdapter",
    "create_uh_model",
    "register_all_uh_models",
    "GammaUH",
    "NashUH",
    "TriangleUH",
    "SequentialFitter",
    "SequentialFitSummary",
]
