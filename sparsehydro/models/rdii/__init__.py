"""sparsehydro.models.rdii — RDII physics models.

Classes
-------
- :class:`~sparsehydro.models.rdii.IAModel` — initial abstraction and rainfall excess
- :class:`~sparsehydro.models.rdii.RTKTriangle` — single triangular RTK unit hydrograph
- :class:`~sparsehydro.models.rdii.RTKEnsembleModel` — additive ensemble of RDIIModels
- :class:`~sparsehydro.models.rdii.RDIIModel` — IA model + N configurable UH components

Helpers
-------
- :func:`~sparsehydro.models.rdii.triangular_uh` — compute RTK triangle ordinate array
- :func:`~sparsehydro.models.rdii.default_rtk_params` — generate initial (R, T, K) tuples
"""

from .initial_abstraction import IAModel
from .rtk_triangle import RTKTriangle, RTKEnsembleModel, triangular_uh, default_rtk_params
from .model import RDIIModel

__all__ = [
    "IAModel",
    "RTKTriangle",
    "RTKEnsembleModel",
    "triangular_uh",
    "default_rtk_params",
    "RDIIModel",
]
