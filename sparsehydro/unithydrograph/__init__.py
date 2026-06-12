"""Unit hydrograph models for sparsehydro.

This subpackage provides:

- :class:`~sparsehydro.unithydrograph.adapter.UnitHydrographAdapter` — the
  adapter base that bridges the existing ``UnitHydrograph`` class to the
  ``IModel`` lifecycle.
- :func:`~sparsehydro.unithydrograph.adapter.create_uh_model` — factory that
  creates a concrete, registry-compatible subclass for any model registered
  in ``UnitHydrograph._registry``.
- :func:`~sparsehydro.unithydrograph.adapter.register_all_uh_models` — bulk
  registration helper.
- :class:`~sparsehydro.unithydrograph.models.GammaUH` — Gamma-function UH
- :class:`~sparsehydro.unithydrograph.models.NashUH` — Nash cascade UH
- :class:`~sparsehydro.unithydrograph.models.TriangleUH` — Triangular UH
- :class:`~sparsehydro.unithydrograph.sequential.SequentialFitter` — sequential event fitting
- :class:`~sparsehydro.unithydrograph.sequential.SequentialFitSummary` — fitting results

Quick start::

    from sparsehydro.unithydrograph import GammaUH, SequentialFitter
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
