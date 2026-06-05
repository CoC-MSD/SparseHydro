"""Adapter that wraps the existing ``UnitHydrograph`` class as an ``IModel``.

The ``UnitHydrograph`` class (from ``uh_models.py``) has its own internal
registry, parameter dict, convolution, and calibration logic.  Rather than
reimplementing any of that, this module bridges it to the sparsehydro
lifecycle and parameter registry via the **Adapter pattern**.

Usage::

    from sparsehydro.unithydrograph import register_all_uh_models

    # Call once after uh_models is on sys.path
    register_all_uh_models()

    from sparsehydro.registry import registry
    model = registry.create("uh-nash")
    model.initialize()
    model.validate()
    model.prepare(rain_stormflow_df)
    result = model.predict()
    model.finalize()

``UnitHydrograph`` is imported **lazily** — ``import sparsehydro`` always
succeeds; the error surfaces only when you first call a method that needs the
wrapped class.  Ensure ``uh_models.py`` and its dependencies
(``data_processing``, ``objective_functions``) are on ``sys.path`` before
calling :func:`register_all_uh_models` or :func:`create_uh_model`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..interfaces import IUnitHydroComponent
from ..parameters import ScalarParameter
from ..registry import registry

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

_UnitHydrograph: type | None = None

_INF_SUBSTITUTE = 1e9  # finite stand-in for np.inf in ScalarParameter bounds


def _get_uh_class() -> type:
    """Return the ``UnitHydrograph`` class, importing it on first call.

    :raises ImportError: If ``uh_models`` is not importable.
    """
    global _UnitHydrograph
    if _UnitHydrograph is None:
        try:
            from uh_models import UnitHydrograph  # type: ignore[import]
            _UnitHydrograph = UnitHydrograph
        except ImportError as exc:
            raise ImportError(
                "uh_models is required for UnitHydrographAdapter. "
                "Ensure uh_models.py and its dependencies "
                "(data_processing, objective_functions) are on sys.path."
            ) from exc
    return _UnitHydrograph


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class UnitHydrographAdapter(IUnitHydroComponent):
    """Adapter base that bridges ``UnitHydrograph`` to the ``IModel`` lifecycle.

    **Do not instantiate or register this class directly.**
    Use :func:`create_uh_model` or :func:`register_all_uh_models` to obtain
    concrete, registry-compatible subclasses.

    **Lifecycle mapping**

    +-----------------+--------------------------------------------------+
    | IModel method   | Adapter action                                   |
    +=================+==================================================+
    | ``initialize``  | Constructs ``UnitHydrograph(key)``, reads        |
    |                 | ``_registry`` to create one ``ScalarParameter``  |
    |                 | per model parameter.                             |
    +-----------------+--------------------------------------------------+
    | ``validate``    | Delegates to ``parameters_valid()``.             |
    +-----------------+--------------------------------------------------+
    | ``prepare``     | Stores the ``rain_stormflow`` DataFrame; syncs   |
    |                 | ``ScalarParameter`` values → ``_uh.parameters``. |
    +-----------------+--------------------------------------------------+
    | ``predict``     | Calls ``_uh.predict()``; returns the DataFrame.  |
    +-----------------+--------------------------------------------------+
    | ``finalize``    | Releases the stored DataFrame.                   |
    +-----------------+--------------------------------------------------+

    **Parameter synchronisation**

    ``ScalarParameter`` values are the source of truth.  They are pushed to
    ``_uh.parameters`` before every ``predict`` or ``fit`` call.  After
    ``fit`` the optimised values are pulled back so both representations stay
    consistent.

    **Infinite bounds**

    ``UnitHydrograph._registry`` uses ``np.inf`` for unbounded parameters
    (e.g. ``A``).  These are clamped to ±:data:`_INF_SUBSTITUTE` when
    creating ``ScalarParameter`` objects so that calibration algorithms that
    require finite bounds always receive them.
    """

    # Placeholder values — overridden by the factory for each concrete class.
    # Keeping them here satisfies IModel.__init_subclass__ for this base.
    model_name: ClassVar[str] = "uh-adapter"
    _uh_model_name: ClassVar[str] = ""
    # All UH models in the registry carry an "A" amplitude/scaling parameter.
    # CombinedHydroModel uses this to exclude A from shape-param re-registration
    # and manage the R fraction separately.
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(self) -> None:
        super().__init__()
        self._uh: Any = None
        self._prepared_data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Construct the wrapped ``UnitHydrograph`` and register its parameters.

        Reads ``UnitHydrograph._registry[_uh_model_name]`` to create one
        :class:`~sparsehydro.parameters.ScalarParameter` per model parameter,
        mapping the registry's ``p0`` / ``lb`` / ``ub`` values directly.

        :raises TypeError: If called on the base ``UnitHydrographAdapter``
            instead of a factory-created subclass.
        """
        UH = _get_uh_class()
        uh_key = type(self)._uh_model_name
        if not uh_key:
            raise TypeError(
                f"{type(self).__name__} has no _uh_model_name set. "
                "Use create_uh_model() or register_all_uh_models() to obtain "
                "a concrete adapter class."
            )

        self._uh = UH(uh_key)
        model_def = UH._registry[uh_key]

        for name, p0, lb, ub in zip(
            model_def["pnames"],
            model_def["p0"],
            model_def["lb"],
            model_def["ub"],
        ):
            lb_f = float(lb) if not np.isinf(lb) else -_INF_SUBSTITUTE
            ub_f = float(ub) if not np.isinf(ub) else _INF_SUBSTITUTE
            self.register_scalar_parameter(
                ScalarParameter(
                    name=name,
                    value=float(p0),
                    lower_bound=lb_f,
                    upper_bound=ub_f,
                )
            )

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate that all registered parameters satisfy their bounds.

        :returns: ``True`` when every parameter is within bounds.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(
        self,
        rain_stormflow: pd.DataFrame,
        **kwargs: Any,
    ) -> None:
        """Store the input DataFrame and sync parameters to the wrapped model.

        :param rain_stormflow: DataFrame with ``datetime``, ``rain``, and
            optionally ``stormflow`` columns — the format expected by
            ``UnitHydrograph.predict`` and ``UnitHydrograph.fit``.
        :type rain_stormflow: pandas.DataFrame
        """
        self._prepared_data = rain_stormflow
        self._sync_to_uh()
        self._state = ModelState.PREPARED

    def predict(
        self,
        predict_range: Any = None,
        trim: bool = True,
    ) -> pd.DataFrame:
        """Convolve stored rainfall with the UH shape and return predicted flow.

        Delegates to ``UnitHydrograph.predict``.

        :param predict_range: Optional time range forwarded to the wrapped
            ``predict`` call.
        :param trim: If ``True``, output is trimmed to the input length.
        :type trim: bool
        :returns: DataFrame with ``datetime`` and ``Q_pred`` columns.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`initialize` or :meth:`prepare` have
            not been called.
        """
        if self._uh is None or self._prepared_data is None:
            raise RuntimeError(
                "Call initialize() and prepare(rain_stormflow) before predict()."
            )
        self._sync_to_uh()
        result: pd.DataFrame = self._uh.predict(
            self._prepared_data, predict_range=predict_range, trim=trim
        )
        self._state = ModelState.PREDICTED
        return result

    def finalize(self) -> None:
        """Release the stored DataFrame and advance state to ``FINALIZED``."""
        self._prepared_data = None
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Calibration passthrough
    # ------------------------------------------------------------------

    def fit(
        self,
        rain_stormflow: pd.DataFrame,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calibrate the wrapped ``UnitHydrograph`` and sync results back.

        All keyword arguments are forwarded to ``UnitHydrograph.fit``
        (``events_list``, ``fit_range``, ``weight_func``, ``method``,
        ``verbose``, etc.).

        After fitting, :class:`~sparsehydro.parameters.ScalarParameter`
        values are updated to reflect the optimised parameters so that
        the sparsehydro registry always reflects the current model state.

        :param rain_stormflow: DataFrame with ``datetime``, ``rain``, and
            ``stormflow`` columns.
        :type rain_stormflow: pandas.DataFrame
        :returns: Dict mapping parameter names to fitted values.
        :rtype: dict[str, float]
        :raises RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._uh is None:
            raise RuntimeError("Call initialize() before fit().")
        self._sync_to_uh()
        fitted: dict[str, float] = self._uh.fit(rain_stormflow, **kwargs)
        self._sync_from_uh()
        return fitted

    # ------------------------------------------------------------------
    # UH shape access
    # ------------------------------------------------------------------

    def get_uh(self, max_steps: int = 864, norm: int = 0) -> np.ndarray:
        """Return the unit hydrograph ordinate array from the wrapped model.

        :param max_steps: Maximum number of time steps for the UH shape.
        :type max_steps: int
        :param norm: If ``1``, return the normalised shape (``A = 1``).
        :type norm: int
        :returns: 1-D array of UH ordinates.
        :rtype: numpy.ndarray
        :raises RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._uh is None:
            raise RuntimeError("Call initialize() before get_uh().")
        self._sync_to_uh()
        return self._uh.get_uh(max_steps=max_steps, norm=norm)

    def get_kernel(
        self,
        dt_hours: float,
        n_steps: int | None = None,
    ) -> np.ndarray:
        """Return the normalized UH ordinate array for use in a composite model.

        Returns ``get_uh(norm=1)`` — the UH shape with the ``A`` amplitude
        parameter divided out, so ``np.sum(result) ≈ 1.0``.

        .. note::
            The UnitHydrograph library generates kernels in native data time
            steps (typically 1 hr).  The ``dt_hours`` argument is accepted for
            API uniformity but does not resample the kernel.  For best results,
            use the same time-step resolution as the data on which the model
            was calibrated.

        :param dt_hours: Time-step size [hr] of the forcing data (informational).
        :param n_steps: If provided, trim or zero-pad the result to this length.
        :returns: 1-D normalised ordinate array.
        :rtype: numpy.ndarray
        :raises RuntimeError: If :meth:`initialize` has not been called.
        """
        if self._uh is None:
            raise RuntimeError("Call initialize() before get_kernel().")
        self._sync_to_uh()
        ordinates = self._uh.get_uh(norm=1)
        if n_steps is not None:
            if len(ordinates) >= n_steps:
                ordinates = ordinates[:n_steps]
            else:
                ordinates = np.pad(ordinates, (0, n_steps - len(ordinates)))
        return ordinates

    @property
    def is_fit(self) -> bool:
        """``True`` if the wrapped ``UnitHydrograph`` has been successfully fitted.

        :rtype: bool
        """
        return bool(self._uh.is_fit) if self._uh is not None else False

    # ------------------------------------------------------------------
    # Parameter synchronisation (private)
    # ------------------------------------------------------------------

    def _sync_to_uh(self) -> None:
        """Push ``ScalarParameter.value`` → ``_uh.parameters`` dict."""
        if self._uh is None:
            return
        for name, param in self._scalar_parameters.items():
            if name in self._uh.parameters:
                self._uh.parameters[name] = param.value

    def _sync_from_uh(self) -> None:
        """Pull ``_uh.parameters`` dict → ``ScalarParameter.value``."""
        if self._uh is None:
            return
        for name, value in self._uh.parameters.items():
            if name in self._scalar_parameters:
                self._scalar_parameters[name].value = float(value)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_uh_model(uh_model_name: str) -> type[UnitHydrographAdapter]:
    """Create and register a concrete ``IModel`` subclass for one UH model.

    The returned class:

    - Has ``model_name = "uh-<slug>"`` where the slug is the lowercased,
      sanitised UH model name (``+`` and ``_`` replaced with ``-``).
    - Is automatically registered in the global sparsehydro
      :data:`~sparsehydro.registry.registry`.
    - Delegates all computation to the existing ``UnitHydrograph``
      implementation without modifying it.

    :param uh_model_name: Key in ``UnitHydrograph._registry``
        (e.g. ``"Nash"``, ``"Gamma"``, ``"Weibull"``).
    :type uh_model_name: str
    :returns: Concrete ``UnitHydrographAdapter`` subclass.
    :rtype: type[UnitHydrographAdapter]
    :raises KeyError: If ``uh_model_name`` is not in
        ``UnitHydrograph._registry``.
    """
    UH = _get_uh_class()
    if uh_model_name not in UH._registry:
        raise KeyError(
            f"'{uh_model_name}' is not in UnitHydrograph._registry. "
            f"Available: {sorted(UH._registry)}"
        )

    slug = uh_model_name.lower().replace("+", "-").replace("_", "-")
    sparse_name = f"uh-{slug}"
    cls_name = f"UH_{uh_model_name.replace('+', '_').replace(' ', '_')}"

    cls = type(
        cls_name,
        (UnitHydrographAdapter,),
        {
            "model_name": sparse_name,
            "_uh_model_name": uh_model_name,
            "__doc__": (
                f"sparsehydro adapter for the ``'{uh_model_name}'`` unit "
                f"hydrograph from ``UnitHydrograph._registry``.\n\n"
                f"Instantiate via ``registry.create('{sparse_name}')``."
            ),
        },
    )

    registry.register(cls)
    return cls


def register_all_uh_models() -> dict[str, type[UnitHydrographAdapter]]:
    """Create and register adapters for **every** model in ``UnitHydrograph._registry``.

    Call this once after confirming that ``uh_models`` is importable::

        import sys
        sys.path.insert(0, "/path/to/UnitHydrograph/Modeling")
        from sparsehydro.unithydrograph import register_all_uh_models
        register_all_uh_models()

    Models that are already registered in the sparsehydro registry are
    skipped silently so the function is safe to call multiple times.

    :returns: Mapping of sparsehydro ``model_name`` → adapter class for
        every model that was registered (new or pre-existing).
    :rtype: dict[str, type[UnitHydrographAdapter]]
    """
    UH = _get_uh_class()
    result: dict[str, type[UnitHydrographAdapter]] = {}
    for uh_name in UH._registry:
        slug = uh_name.lower().replace("+", "-").replace("_", "-")
        sparse_name = f"uh-{slug}"
        if registry.is_registered(sparse_name):
            result[sparse_name] = registry.get(sparse_name)  # type: ignore[assignment]
        else:
            result[sparse_name] = create_uh_model(uh_name)
    return result
