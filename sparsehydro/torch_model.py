"""PyTorch-compatible model interface for gradient-based parameter estimation.

:class:`ITorchModel` combines the sparsehydro :class:`~sparsehydro.interfaces.IModel`
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

from .enums import ModelState
from .interfaces import IModel


try:
    import torch
    import torch.nn as nn

    class ITorchModel(IModel, nn.Module):
        """Abstract interface for differentiable parsimonious models.

        Inherits from both :class:`~sparsehydro.interfaces.IModel` and
        ``torch.nn.Module``.  Subclasses must implement all five
        :class:`~sparsehydro.interfaces.IModel` lifecycle methods and
        :meth:`forward` — the differentiable computation graph.

        The default :meth:`predict` implementation calls :meth:`forward` and
        advances model state to ``PREDICTED``.  Override :meth:`predict` if
        additional pre/post-processing around :meth:`forward` is required.

        Gradient-tracked parameters should be declared as ``torch.nn.Parameter``
        attributes so that PyTorch optimisers can discover them via
        ``Module.parameters()``.  The sparsehydro
        :meth:`~sparsehydro.interfaces.IModel.register_scalar_parameter` /
        :meth:`~sparsehydro.interfaces.IModel.register_vector_parameter`
        registry holds *metadata* (bounds, units, description) for the same
        parameters to allow calibration algorithms to apply constraints.

        Example::

            import torch
            import torch.nn as nn
            from sparsehydro import ModelState, ScalarParameter
            from sparsehydro.torch_model import ITorchModel

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
            """Differentiable forward pass.

            Implement the model computation using PyTorch operations so that
            gradients propagate back through all ``nn.Parameter`` attributes.
            Called by the default :meth:`predict` implementation.

            :returns: Output tensor produced by the model.
            :rtype: torch.Tensor
            """

        def predict(self, *args: Any, **kwargs: Any) -> "torch.Tensor":
            """Run the differentiable forward pass and return the output tensor.

            Delegates to :meth:`forward` and advances model state to
            ``PREDICTED``.

            :returns: Output tensor from :meth:`forward`.
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
