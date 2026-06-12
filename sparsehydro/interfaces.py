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

import numpy as np
import pandas as pd

from .enums import ModelState
from .parameters import ConstraintRecord, FieldRecord, ScalarParameter, VectorParameter


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
        self._constraint_registry: list[ConstraintRecord] = []
        self._output_fields: dict[str, FieldRecord] = {}

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

    def rename_scalar_parameter(self, old_name: str, new_name: str) -> None:
        """Rename a registered scalar parameter.

        Updates both the registry key and the parameter's ``name`` field.
        The parameter object itself is unchanged apart from its name.

        :param old_name: Current parameter name.
        :type old_name: str
        :param new_name: Desired new name.
        :type new_name: str
        :raises KeyError: If ``old_name`` is not registered.
        :raises ValueError: If ``new_name`` is already registered.
        """
        if new_name in self._scalar_parameters:
            raise ValueError(f"Scalar parameter '{new_name}' already exists.")
        param = self.get_scalar_parameter(old_name)
        param.name = new_name
        self._scalar_parameters[new_name] = param
        del self._scalar_parameters[old_name]

    def rename_vector_parameter(self, old_name: str, new_name: str) -> None:
        """Rename a registered vector parameter.

        Updates both the registry key and the parameter's ``name`` field.

        :param old_name: Current parameter name.
        :type old_name: str
        :param new_name: Desired new name.
        :type new_name: str
        :raises KeyError: If ``old_name`` is not registered.
        :raises ValueError: If ``new_name`` is already registered.
        """
        if new_name in self._vector_parameters:
            raise ValueError(f"Vector parameter '{new_name}' already exists.")
        param = self.get_vector_parameter(old_name)
        param.name = new_name
        self._vector_parameters[new_name] = param
        del self._vector_parameters[old_name]

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

    # ------------------------------------------------------------------
    # Output field registry
    # ------------------------------------------------------------------

    def register_output_field(self, field: FieldRecord) -> None:
        """Register metadata for one column of the ``predict()`` output DataFrame.

        Call inside :meth:`initialize` for every column your model writes so
        that calibration tooling and notebooks can introspect the output
        schema.  Registering ``datetime`` is encouraged even though it carries
        no physical units.

        If a field with the same name is already registered it is overwritten.

        :param field: Column metadata.
        :type field: FieldRecord
        """
        self._output_fields[field.name] = field

    def get_output_field(self, name: str) -> FieldRecord:
        """Return the :class:`~sparsehydro.parameters.FieldRecord` for *name*.

        :param name: Column name as registered.
        :type name: str
        :returns: The matching field record.
        :rtype: FieldRecord
        :raises KeyError: If *name* has not been registered.
        """
        if name not in self._output_fields:
            available = list(self._output_fields)
            raise KeyError(
                f"Output field '{name}' not found. "
                f"Available fields: {available}"
            )
        return self._output_fields[name]

    @property
    def output_field_names(self) -> list[str]:
        """Ordered list of registered output column names.

        :rtype: list[str]
        """
        return list(self._output_fields)

    @property
    def output_fields(self) -> list[FieldRecord]:
        """Ordered list of all registered :class:`~sparsehydro.parameters.FieldRecord` objects.

        :rtype: list[FieldRecord]
        """
        return list(self._output_fields.values())

    @property
    def calibratable_output_names(self) -> list[str]:
        """Names of output columns flagged as calibratable (``calibratable=True``).

        These are the columns suitable for use as the ``predicted`` target in a
        :class:`~sparsehydro.calibration.CalibrationProblem`.

        :rtype: list[str]
        """
        return [f.name for f in self._output_fields.values() if f.calibratable]

    def output_field_metadata(self) -> "pd.DataFrame":
        """Return a DataFrame describing every registered output field.

        Columns: ``field``, ``units``, ``calibratable``, ``description``.

        Suitable for :func:`display` in Jupyter notebooks::

            display(model.output_field_metadata())

        :returns: One row per registered output field.
        :rtype: pandas.DataFrame
        """
        rows = [
            {
                "field":        f.name,
                "units":        f.units,
                "calibratable": f.calibratable,
                "description":  f.description,
            }
            for f in self._output_fields.values()
        ]
        return pd.DataFrame(rows)

    def register_inequality_constraint(self, record: ConstraintRecord) -> None:
        """Register a named inequality constraint.

        Call this inside :meth:`initialize` after registering the parameters
        the constraint acts upon.  The order of registration must match the
        index order of the residuals returned by :meth:`inequality_constraints`.

        :param record: Constraint metadata (name + description).
        :type record: ConstraintRecord
        """
        self._constraint_registry.append(record)

    @property
    def inequality_constraint_names(self) -> list[str]:
        """Ordered list of registered inequality constraint names.

        :rtype: list[str]
        """
        return [r.name for r in self._constraint_registry]

    @property
    def inequality_constraint_descriptions(self) -> list[str]:
        """Ordered list of registered inequality constraint descriptions.

        :rtype: list[str]
        """
        return [r.description for r in self._constraint_registry]

    def inequality_constraints(self) -> list[float]:
        """Inequality constraint residuals where ``g_j <= 0`` means feasible.

        Override in subclasses to expose model-specific constraints to the
        calibration framework.  The default returns an empty list (unconstrained).
        Register corresponding metadata via :meth:`register_inequality_constraint`
        so that solvers and notebooks can display meaningful labels.

        :returns: List of constraint residuals.
        :rtype: list[float]
        """
        return []


class IUnitHydroComponent(IModel, ABC):
    """Abstract base for unit hydrograph components used within :class:`CombinedHydroModel`.

    Extends :class:`IModel` with a single additional method, :meth:`get_kernel`,
    that returns a normalized ordinate array suitable for convolution with a
    rainfall-excess series.

    **Amplitude parameter**

    Many UH models carry a parameter that scales the total response volume
    (``R`` for RTK triangles, ``A`` for Nash/Gamma UH shapes).  When a
    component is embedded in :class:`CombinedHydroModel`, that composite
    manages the fraction ``R_i`` separately and re-registers only the
    *shape* parameters.  Declare the name of the amplitude/scaling
    parameter as a class variable so the composite can exclude it::

        class MyUH(IUnitHydroComponent):
            _amplitude_param_name = "A"   # or "R", or None if none
    """

    _amplitude_param_name: ClassVar[str | None] = None

    @abstractmethod
    def get_kernel(
        self,
        dt_hours: float,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """Return normalized unit hydrograph ordinates.

        The returned array satisfies::

            np.sum(ordinates) * dt_hours ≈ 1.0

        Multiply element-wise by ``R * p_excess_mm`` and sum across components
        to obtain the RDII depth time series.

        :param dt_hours: Time-step size [hr] of the forcing data.
        :type dt_hours: float
        :param n_steps: If provided, trim or zero-pad the result to this
            length.  Defaults to the natural support of the kernel.
        :type n_steps: int, optional
        :returns: 1-D ordinate array [1/hr], length ``n_steps`` or natural.
        :rtype: numpy.ndarray
        """
