"""PyTorch-compatible model interface for gradient-based parameter estimation.

:class:`ITorchModel` combines the sparsehydro :class:`~sparsehydro.models.IModel`
lifecycle with ``torch.nn.Module``, enabling automatic differentiation through the
model computation so that parameters can be optimised with gradient-based methods
(e.g. Adam, L-BFGS).

PyTorch is an *optional* dependency.  If it is not installed, importing this
module defines a placeholder class that raises :class:`ImportError` on
instantiation, preserving the package's ability to run without torch.

Install with::

    pip install sparsehydro[torch]
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ..enums import ModelState
from .base import IModel


try:
    import torch
    import torch.nn as nn

    class ITorchModel(IModel, nn.Module):
        """Abstract interface for differentiable parsimonious models.

        Inherits from both :class:`~sparsehydro.models.IModel` and
        ``torch.nn.Module``.  Subclasses must implement all five
        :class:`~sparsehydro.models.IModel` lifecycle methods and
        :meth:`forward` — the differentiable computation graph.

        The default :meth:`predict` implementation calls :meth:`forward` and
        advances model state to ``PREDICTED``.  Override :meth:`predict` if
        additional pre/post-processing around :meth:`forward` is required.

        Gradient-tracked parameters should be declared as ``torch.nn.Parameter``
        attributes so that PyTorch optimisers can discover them via
        ``Module.parameters()``.  The sparsehydro
        :meth:`~sparsehydro.models.IModel.register_scalar_parameter` /
        :meth:`~sparsehydro.models.IModel.register_vector_parameter`
        registry holds *metadata* (bounds, units, description) for the same
        parameters to allow calibration algorithms to apply constraints.

        Example::

            import torch
            import torch.nn as nn
            from sparsehydro import ModelState, ScalarParameter
            from sparsehydro.models import ITorchModel

            class LinearReservoir(ITorchModel):
                def initialize(self) -> None:
                    self.k = nn.Parameter(torch.tensor(0.5))
                    self.register_scalar_parameter(
                        ScalarParameter("k", value=0.5,
                                        lower_bound=0.0, upper_bound=1.0)
                    )
                    self._state = ModelState.INITIALIZED

                def validate(self) -> bool:
                    ok = self.parameters_valid()
                    if ok:
                        self._state = ModelState.VALIDATED
                    return ok

                def prepare(self, forcing: torch.Tensor) -> None:
                    self._forcing = forcing
                    self._state = ModelState.PREPARED

                def forward(self, forcing: torch.Tensor) -> torch.Tensor:
                    return self.k * forcing

                def finalize(self) -> None:
                    self._state = ModelState.FINALIZED
        """

        def __init__(self) -> None:
            IModel.__init__(self)
            nn.Module.__init__(self)

        @abstractmethod
        def forward(self, *args: Any, **kwargs: Any) -> "torch.Tensor":
            """Differentiable forward pass producing the model output.

            Subclasses implement the model computation using torch operations
            so that gradients flow back to the registered ``torch.nn.Parameter``
            attributes.

            :param args: Positional forcing inputs (e.g. a rainfall tensor).
            :type args: Any
            :param kwargs: Keyword forcing inputs.
            :type kwargs: Any
            :returns: The model output tensor with gradient tracking enabled.
            :rtype: torch.Tensor
            """

        def predict(self, *args: Any, **kwargs: Any) -> "torch.Tensor":
            """Run the differentiable forward pass and advance model state.

            Delegates to :meth:`forward` and sets the model state to
            :attr:`~sparsehydro.enums.ModelState.PREDICTED`.

            :param args: Positional forcing inputs forwarded to :meth:`forward`.
            :type args: Any
            :param kwargs: Keyword forcing inputs forwarded to :meth:`forward`.
            :type kwargs: Any
            :returns: The output tensor returned by :meth:`forward`.
            :rtype: torch.Tensor
            """
            result = self.forward(*args, **kwargs)
            self._state = ModelState.PREDICTED
            return result

except ImportError:  # pragma: no cover

    class ITorchModel:  # type: ignore[no-redef]
        """Placeholder raised when PyTorch is not installed.

        Install PyTorch to use this class::

            pip install sparsehydro[torch]

        :raises ImportError: Always raised on instantiation.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required to use ITorchModel. "
                "Install it with:  pip install sparsehydro[torch]"
            )
