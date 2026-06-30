"""sparsehydro.models — model interfaces, generic models, and domain sub-packages.

Sub-packages
------------
- :mod:`sparsehydro.models.rdii` — RDII physics models (IAModel, RDIIModel, RTKTriangle)
- :mod:`sparsehydro.models.unithydrograph` — unit hydrograph adapters and implementations

Interfaces
----------
- :class:`~sparsehydro.models.IModel` — abstract lifecycle interface for all models
- :class:`~sparsehydro.models.IUnitHydroComponent` — extension for UH kernel models
- :class:`~sparsehydro.models.ITorchModel` — optional PyTorch-compatible interface

Generic models
--------------
- :class:`~sparsehydro.models.EnsembleModel` — weighted multi-model compositor
- :class:`~sparsehydro.models.SeasonalityModel` — discrete peaking-factor seasonal flow model
"""

from .base import IModel, IUnitHydroComponent
from .amm import AMMModel
from .ensemble import EnsembleModel
from .seasonality import SeasonalityModel

__all__ = [
    "IModel",
    "IUnitHydroComponent",
    "AMMModel",
    "EnsembleModel",
    "SeasonalityModel",
    "ITorchModel",
]

try:
    from .torch_model import ITorchModel
except ImportError:  # pragma: no cover
    pass
