"""Abstract base interface for parsimonious hydrological models.

All concrete models must inherit from :class:`IModel` and implement the five
lifecycle methods.  The class also provides a built-in parameter registry
so that calibration algorithms can discover and manipulate named parameters
without knowing the internal model structure.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from .enums import ModelState
from .parameters import ScalarParameter, VectorParameter


class IModel(ABC):
    """Abstract interface for a parsimonious hydrological model.

    **Model name**

    Every concrete subclass must define a ``model_name`` class variable — a
    non-empty string that uniquely identifies the model type.  This is
    enforced at class-definition time::

        class MyModel(IModel):
            model_name = "my-model"
            ...

    Attempting to define a concrete subclass without ``model_name`` raises
    :class:`TypeError` immediately.

    **Lifecycle**

    Concrete subclasses must advance ``self._state`` as each method completes::

        CREATED → INITIALIZED → VALIDATED → PREPARED → PREDICTED → FINALIZED

    **Parameter registry**

    Parameters are registered during :meth:`initialize` via
    :meth:`register_scalar_parameter` and :meth:`register_vector_parameter`.
    Calibration tools retrieve them by name through
    :meth:`get_scalar_parameter` and :meth:`get_vector_parameter`.

    Minimal concrete implementation::

        class MyModel(IModel):
            model_name = "my-model"

            def initialize(self) -> None:
                self.register_scalar_parameter(
                    ScalarParameter("alpha", value=0.3,
                                    lower_bound=0.0, upper_bound=1.0)
                )
                self._state = ModelState.INITIALIZED

            def validate(self) -> bool:
                ok = self.parameters_valid()
                if ok:
                    self._state = ModelState.VALIDATED
                return ok

            def prepare(self, forcing) -> None:
                self._forcing = forcing
                self._state = ModelState.PREPARED

            def predict(self) -> pd.Series:
                alpha = self.get_scalar_parameter("alpha").value
                self._state = ModelState.PREDICTED
                return pd.Series(self._forcing * alpha)

            def finalize(self) -> None:
                self._state = ModelState.FINALIZED
    """

    #: Unique string identifier for this model type.
    #: Must be defined by every concrete subclass.
    model_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract subclasses (those that still have unimplemented abstract
        # methods) are permitted to defer the model_name definition.
        if inspect.isabstract(cls):
            return
        if "model_name" not in vars(cls) and not hasattr(cls, "model_name"):
            raise TypeError(
                f"Concrete model '{cls.__qualname__}' must define a "
                "'model_name' class variable (a non-empty string that "
                "uniquely identifies the model type)."
            )
        name = getattr(cls, "model_name", None)
        if not isinstance(name, str) or not name.strip():
            raise TypeError(
                f"'{cls.__qualname__}.model_name' must be a non-empty string; "
                f"got {name!r}."
            )

    def __init__(self) -> None:
        self._state: ModelState = ModelState.CREATED
        self._scalar_parameters: dict[str, ScalarParameter] = {}
        self._vector_parameters: dict[str, VectorParameter] = {}

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> ModelState:
        """Current lifecycle state of the model.

        :rtype: ModelState
        """
        return self._state

    def is_created(self) -> bool:
        """Return ``True`` when the model is in the ``CREATED`` state.

        :rtype: bool
        """
        return self._state is ModelState.CREATED

    def is_initialized(self) -> bool:
        """Return ``True`` when the model is in the ``INITIALIZED`` state.

        :rtype: bool
        """
        return self._state is ModelState.INITIALIZED

    def is_validated(self) -> bool:
        """Return ``True`` when the model is in the ``VALIDATED`` state.

        :rtype: bool
        """
        return self._state is ModelState.VALIDATED

    def is_prepared(self) -> bool:
        """Return ``True`` when the model is in the ``PREPARED`` state.

        :rtype: bool
        """
        return self._state is ModelState.PREPARED

    def is_predicted(self) -> bool:
        """Return ``True`` when the model is in the ``PREDICTED`` state.

        :rtype: bool
        """
        return self._state is ModelState.PREDICTED

    def is_finalized(self) -> bool:
        """Return ``True`` when the model is in the ``FINALIZED`` state.

        :rtype: bool
        """
        return self._state is ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Lifecycle (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Set up model structure and register parameters.

        Implementations should call :meth:`register_scalar_parameter` and/or
        :meth:`register_vector_parameter` for every calibratable parameter,
        then set ``self._state = ModelState.INITIALIZED``.
        """

    @abstractmethod
    def validate(self) -> bool:
        """Validate the model configuration and parameter bounds.

        Implementations should call :meth:`parameters_valid` as part of
        validation and advance ``self._state = ModelState.VALIDATED`` only
        when returning ``True``.

        :returns: ``True`` if the model is ready to run; ``False`` otherwise.
        :rtype: bool
        """

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> None:
        """Load and pre-process forcing/input data.

        Should set ``self._state = ModelState.PREPARED`` on success.
        """

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame | pd.Series:
        """Execute the model and return outputs as a DataFrame or Series.

        Should set ``self._state = ModelState.PREDICTED`` on success.

        :returns: Model outputs as a pandas DataFrame or Series.
        :rtype: pandas.DataFrame or pandas.Series
        """

    @abstractmethod
    def finalize(self) -> None:
        """Release resources and wrap up computation.

        Should set ``self._state = ModelState.FINALIZED``.
        """

    # ------------------------------------------------------------------
    # Parameter registry
    # ------------------------------------------------------------------

    def register_scalar_parameter(self, param: ScalarParameter) -> None:
        """Register a scalar parameter with the model.

        If a parameter with the same name already exists it is overwritten.

        :param param: The scalar parameter to register.
        :type param: ScalarParameter
        """
        self._scalar_parameters[param.name] = param

    def register_vector_parameter(self, param: VectorParameter) -> None:
        """Register a vector parameter with the model.

        If a parameter with the same name already exists it is overwritten.

        :param param: The vector parameter to register.
        :type param: VectorParameter
        """
        self._vector_parameters[param.name] = param

    def get_scalar_parameter(self, name: str) -> ScalarParameter:
        """Retrieve a registered scalar parameter by name.

        :param name: The parameter name used when it was registered.
        :type name: str
        :returns: The matching scalar parameter.
        :rtype: ScalarParameter
        :raises KeyError: If no scalar parameter with ``name`` has been
            registered.
        """
        if name not in self._scalar_parameters:
            available = list(self._scalar_parameters)
            raise KeyError(
                f"Scalar parameter '{name}' not found. "
                f"Available parameters: {available}"
            )
        return self._scalar_parameters[name]

    def get_vector_parameter(self, name: str) -> VectorParameter:
        """Retrieve a registered vector parameter by name.

        :param name: The parameter name used when it was registered.
        :type name: str
        :returns: The matching vector parameter.
        :rtype: VectorParameter
        :raises KeyError: If no vector parameter with ``name`` has been
            registered.
        """
        if name not in self._vector_parameters:
            available = list(self._vector_parameters)
            raise KeyError(
                f"Vector parameter '{name}' not found. "
                f"Available parameters: {available}"
            )
        return self._vector_parameters[name]

    @property
    def scalar_parameter_names(self) -> list[str]:
        """Ordered list of registered scalar parameter names.

        :rtype: list[str]
        """
        return list(self._scalar_parameters)

    @property
    def vector_parameter_names(self) -> list[str]:
        """Ordered list of registered vector parameter names.

        :rtype: list[str]
        """
        return list(self._vector_parameters)

    def parameters_valid(self) -> bool:
        """Return ``True`` if every registered parameter is within its bounds.

        Checks both scalar and vector parameters.

        :returns: ``True`` when all parameters satisfy their bounds constraints.
        :rtype: bool
        """
        return all(p.is_valid() for p in self._scalar_parameters.values()) and all(
            p.is_valid() for p in self._vector_parameters.values()
        )
