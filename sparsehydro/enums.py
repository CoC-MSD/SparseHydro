"""Enumerations for tracking model lifecycle state in sparsehydro."""

from enum import Enum


class ModelState(Enum):
    """Enumeration of the six lifecycle states for a parsimonious model.

    States follow a strict progression::

        CREATED → INITIALIZED → VALIDATED → PREPARED → PREDICTED → FINALIZED

    Concrete :class:`~sparsehydro.models.IModel` implementations are
    responsible for advancing ``_state`` as each lifecycle method completes.

    :cvar CREATED: Object has been instantiated; no parameters registered yet.
    :cvar INITIALIZED: ``initialize()`` has completed; parameters are
        registered and model structure is set up.
    :cvar VALIDATED: ``validate()`` returned ``True``; all parameters are
        within their specified bounds.
    :cvar PREPARED: ``prepare()`` has completed; forcing/input data are loaded
        and pre-processed.
    :cvar PREDICTED: ``predict()`` has completed; model outputs are available.
    :cvar FINALIZED: ``finalize()`` has completed; resources have been released.
    """

    CREATED = "created"
    INITIALIZED = "initialized"
    VALIDATED = "validated"
    PREPARED = "prepared"
    PREDICTED = "predicted"
    FINALIZED = "finalized"

    def __str__(self) -> str:  # pragma: no cover
        return self.value
