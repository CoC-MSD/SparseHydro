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

Quick start::

    import sys
    sys.path.insert(0, "/path/to/UnitHydrograph/Modeling")

    from sparsehydro.unithydrograph import register_all_uh_models
    from sparsehydro.registry import registry

    register_all_uh_models()

    model = registry.create("uh-nash")
    model.initialize()
    model.validate()
    model.prepare(rain_df)
    result = model.predict()
    model.finalize()
"""

from .adapter import (
    UnitHydrographAdapter,
    create_uh_model,
    register_all_uh_models,
)

__all__ = [
    "UnitHydrographAdapter",
    "create_uh_model",
    "register_all_uh_models",
]
