"""CalibrationProblem: wraps any IModel with data, observed targets, and objectives.

All column/data mappings — including the observed target and the predicted output —
are specified through a single ``column_map`` dictionary.  Values can be either a
**column name string** (for DataFrame data) or a **callable** (for any data type).

**DataFrame workflow** (most common)::

    from sparsehydro.calibration import CalibrationProblem, NashSutcliffe, PeakWeightedMSE
    from sparsehydro.models.rdii import RDIIModel

    model = RDIIModel(n_triangles=3)
    model.initialize()
    model.validate()

    problem = CalibrationProblem(
        model=model,
        data=df,
        objectives=[PeakWeightedMSE(), NashSutcliffe()],
        column_map={
            # Data prep: target_col → source_col (rename before model.prepare)
            "rainfall_mm":   "rain",
            "temperature_c": "air_temp_c",
            # Reserved roles (string = column name)
            "observed":  "stormflow",  # column in prepared data → target array
            "predicted": "rdii_cfs",   # column in model.predict() output
        },
    )
    result = NSGAIISolver().solve(problem)

**Generic / callable workflow**::

    problem = CalibrationProblem(
        model=model,
        data=raw_dict,
        objectives=[PeakWeightedMSE()],
        prepare_fn=lambda d, **kw: build_dataframe(d, **kw),
        column_map={
            # Reserved roles (callable = extractor function)
            "observed":  lambda prepared: prepared["obs"].to_numpy(),
            "predicted": lambda pred_df: pred_df["rdii_cfs"].to_numpy(),
        },
    )
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np
import pandas as pd

from .objectives import IObjective

if TYPE_CHECKING:
    from ..models import IModel

# Reserved keys in column_map that specify calibration roles, not column renames.
_ROLE_KEYS = frozenset({"observed", "predicted", "mask"})


def _resolve_extractor(
    mapping_value: Union[str, Callable, None],
    role: str,
) -> Callable:
    """Return a callable extractor from a column_map value.

    :param mapping_value: Column name (str) or callable, or None.
    :type mapping_value: str or Callable or None
    :param role: Role name for error messages (``"observed"`` / ``"predicted"``).
    :type role: str
    :returns: Callable ``(data) → np.ndarray``.
    :rtype: Callable
    :raises ValueError: If *mapping_value* is ``None``.
    :raises TypeError: If *mapping_value* is neither a string nor callable.
    """
    if mapping_value is None:
        raise ValueError(
            f"column_map['{role}'] is required.  "
            f"Supply a column name (str) or a callable ``(data) -> np.ndarray``."
        )
    if callable(mapping_value):
        return mapping_value
    if isinstance(mapping_value, str):
        col = mapping_value
        return lambda data, c=col: (
            data[c].to_numpy(dtype=float)
            if isinstance(data, pd.DataFrame)
            else np.asarray(data[c], dtype=float)
        )
    raise TypeError(
        f"column_map['{role}'] must be a column name (str) or callable; "
        f"got {type(mapping_value).__name__!r}."
    )


def _resolve_mask(
    spec: Union[str, Callable, np.ndarray, None],
    prepared_data: Any,
    length: int,
) -> np.ndarray | None:
    """Resolve a mask specification into a boolean array.

    :param spec: ``None``, a boolean ``np.ndarray``, a column-name string, or a
        callable ``(prepared_data) -> array-like``.
    :type spec: str or Callable or numpy.ndarray or None
    :param prepared_data: Prepared data used to resolve string/callable specs.
    :type prepared_data: Any
    :param length: Expected mask length (matches the observed array).
    :type length: int
    :returns: Boolean array of length *length*, or ``None`` if *spec* is ``None``.
    :rtype: numpy.ndarray or None
    :raises ValueError: If the resolved mask length does not match *length*.
    """
    if spec is None:
        return None
    if isinstance(spec, np.ndarray):
        arr = spec
    else:
        extractor = _resolve_extractor(spec, "mask")
        arr = extractor(prepared_data)
    mask = np.asarray(arr, dtype=bool)
    if mask.shape != (length,):
        raise ValueError(
            f"mask length {mask.shape} does not match observed length ({length},)."
        )
    return mask


class CalibrationProblem:
    """Solver-agnostic calibration problem for any :class:`~sparsehydro.models.IModel`.

    Bundles a model with input data, preprocessing, observed targets, and objectives.
    The same problem instance can be passed to any
    :class:`~sparsehydro.calibration.solvers.ISolver` without modification.

    All column/role mappings are expressed through a single ``column_map`` dict:

    * Non-reserved entries ``{target_col: source_col}`` — rename ``data[source_col]``
      to ``target_col`` before calling ``model.prepare()``.
    * Reserved key ``"observed"`` — **str** (column name) or **callable**
      ``(prepared_data) → np.ndarray``.  Identifies the calibration target.
    * Reserved key ``"predicted"`` — **str** or **callable**
      ``(predict_df) → np.ndarray``.  Identifies the model output to compare.

    :param model: Model in at least VALIDATED state.  Prepared automatically when
        ``data`` is supplied.
    :type model: IModel
    :param data: Input data (any type).  Passed through the preprocessing pipeline
        and then to ``model.prepare()``.  May be ``None`` if the model is already
        in PREPARED state, but then ``column_map["observed"]`` must be a callable
        that does not require the prepared data (e.g. a closure).
    :type data: Any
    :param objectives: At least one objective.
    :type objectives: list[IObjective]
    :param column_map: Unified mapping dict (see above).
    :type column_map: dict[str, Any] or None
    :param prepare_fn: ``(data, **prepare_kwargs) → prepared_data`` callable for
        complex preprocessing.  Mutually exclusive with non-reserved ``column_map``
        rename entries.
    :type prepare_fn: Callable or None
    :param mask: Problem-level default mask restricting every objective to a
        subset of timesteps.  May be a boolean ``np.ndarray``, a column-name
        string, or a callable ``(prepared_data) -> array-like`` — resolved like
        ``observed``/``predicted``.  Equivalent to ``column_map["mask"]`` (the
        explicit argument wins if both are given).  Individual objectives that
        carry their own ``mask`` override this default.  Positions where
        observed or predicted is ``NaN`` are always excluded regardless.
    :type mask: str or Callable or numpy.ndarray or None
    :param prepare_kwargs: Extra keyword arguments forwarded to ``prepare_fn``.
    :type prepare_kwargs: Any

    **Masking**::

        problem = CalibrationProblem(
            model=model, data=df,
            objectives=[NashSutcliffe(), VolumeRelativeError()],
            column_map={"observed": "stormflow", "predicted": "rdii_cfs",
                        "mask": "is_storm"},   # bool column in prepared data
        )
    """

    def __init__(
        self,
        model: "IModel",
        data: Any = None,
        objectives: list[IObjective] | None = None,
        column_map: "dict[str, Any] | None" = None,
        prepare_fn: Callable | None = None,
        mask: "str | Callable | np.ndarray | None" = None,
        **prepare_kwargs: Any,
    ) -> None:
        if objectives is None or len(objectives) == 0:
            raise ValueError("At least one objective is required.")

        cmap = column_map or {}
        rename_entries = {k: v for k, v in cmap.items() if k not in _ROLE_KEYS}
        if rename_entries and prepare_fn is not None:
            raise ValueError(
                "column_map rename entries and prepare_fn are mutually exclusive. "
                "Use column_map renames for simple DataFrame column renaming; "
                "use prepare_fn for complex or non-DataFrame preprocessing."
            )

        self._model = model
        self._objectives = list(objectives)
        self._column_map = cmap
        self._prepare_fn = prepare_fn
        self._prepare_kwargs = prepare_kwargs

        # ------------------------------------------------------------------
        # Resolve extractors from column_map
        # ------------------------------------------------------------------
        self._result_extractor: Callable = _resolve_extractor(
            cmap.get("predicted"), "predicted"
        )
        observed_extractor: Callable = _resolve_extractor(
            cmap.get("observed"), "observed"
        )

        # ------------------------------------------------------------------
        # Preprocess data and prepare model
        # ------------------------------------------------------------------
        if data is not None:
            prepared_data = self._preprocess(data)
            if not model.is_prepared():
                model.prepare(prepared_data)
        elif not model.is_prepared():
            raise RuntimeError(
                "model must be in PREPARED state, or supply data= so the "
                "problem can call model.prepare() automatically."
            )
        else:
            prepared_data = None

        # ------------------------------------------------------------------
        # Extract observed array
        # ------------------------------------------------------------------
        try:
            self._observed: np.ndarray = np.asarray(
                observed_extractor(prepared_data), dtype=float
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to extract observed values from data using "
                f"column_map['observed']={cmap.get('observed')!r}: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Resolve masks (problem-level default + per-objective overrides)
        # ------------------------------------------------------------------
        n_obs = int(self._observed.shape[0])
        mask_spec = mask if mask is not None else cmap.get("mask")
        try:
            self._mask: np.ndarray | None = _resolve_mask(
                mask_spec, prepared_data, n_obs
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to resolve problem-level mask {mask_spec!r}: {exc}"
            ) from exc

        self._objective_masks: list[np.ndarray | None] = [
            _resolve_mask(obj.mask, prepared_data, n_obs) for obj in self._objectives
        ]

        # ------------------------------------------------------------------
        # Parameter registry snapshot (calibrate=True only)
        # ------------------------------------------------------------------
        self._param_names = [
            n for n in model.scalar_parameter_names
            if model.get_scalar_parameter(n).calibrate
        ]
        self._xl = np.array(
            [model.get_scalar_parameter(n).lower_bound for n in self._param_names],
            dtype=float,
        )
        self._xu = np.array(
            [model.get_scalar_parameter(n).upper_bound for n in self._param_names],
            dtype=float,
        )
        self._n_ieq_constr: int = len(model.inequality_constraints())
        self._constraint_names: list[str] = list(model.inequality_constraint_names)
        self._constraint_descriptions: list[str] = list(model.inequality_constraint_descriptions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, data: Any) -> Any:
        """Apply the data pipeline (prepare_fn or column_map renaming)."""
        if self._prepare_fn is not None:
            return self._prepare_fn(data, **self._prepare_kwargs)

        rename_map = {
            v: k for k, v in self._column_map.items()
            if k not in _ROLE_KEYS and isinstance(v, str)
        }
        if rename_map and isinstance(data, pd.DataFrame):
            data = data.rename(columns=rename_map)
        return data

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Number of calibratable scalar parameters."""
        return len(self._param_names)

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self._objectives)

    @property
    def param_names(self) -> list[str]:
        """Ordered list of calibrated parameter names."""
        return list(self._param_names)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds as ``(xl, xu)`` arrays."""
        return self._xl.copy(), self._xu.copy()

    @property
    def objective_names(self) -> list[str]:
        """Objective names in evaluation order."""
        return [obj.name for obj in self._objectives]

    @property
    def minimize_flags(self) -> list[bool]:
        """``True`` if the corresponding objective should be minimised."""
        return [obj.minimize for obj in self._objectives]

    @property
    def n_ieq_constr(self) -> int:
        """Number of inequality constraints reported by the model."""
        return self._n_ieq_constr

    @property
    def constraint_names(self) -> list[str]:
        """Ordered list of inequality constraint names (empty if unconstrained)."""
        return list(self._constraint_names)

    @property
    def constraint_descriptions(self) -> list[str]:
        """Ordered list of inequality constraint descriptions."""
        return list(self._constraint_descriptions)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray, penalty_weight: float = 1e6) -> np.ndarray:
        """Apply a parameter vector, run ``predict()``, and return objectives.

        Objective values are in **minimisation form**: maximised objectives are
        negated.  When ``penalty_weight > 0`` and the model reports inequality
        constraints, a squared penalty ``penalty_weight * Σ max(0, g_j)²`` is
        added to steer constraint-unaware solvers toward feasible regions.
        Pass ``penalty_weight=0.0`` when the solver handles constraints natively.

        Only parameters with ``calibrate=True`` are set from *x*; fixed
        parameters (``calibrate=False``) retain their current values.

        :param x: Parameter vector of length :attr:`n_params` (calibratable only).
        :type x: numpy.ndarray
        :param penalty_weight: Multiplier for the squared-penalty term (default 1e6).
        :type penalty_weight: float
        :returns: 1-D array of length :attr:`n_objectives` in minimisation form.
        :rtype: numpy.ndarray
        """
        for name, val in zip(self._param_names, x):
            self._model.get_scalar_parameter(name).value = float(val)

        pred_df = self._model.predict()
        predicted = self._result_extractor(pred_df)

        F = np.empty(len(self._objectives), dtype=float)
        for i, obj in enumerate(self._objectives):
            eff = self._objective_masks[i]
            if eff is None:
                eff = self._mask
            try:
                val = obj.compute(self._observed, predicted, mask=eff)
            except Exception:
                val = 1e12 if obj.minimize else -1e12
            F[i] = val if obj.minimize else -val

        if penalty_weight > 0.0 and self._n_ieq_constr > 0:
            G = np.array(self._model.inequality_constraints(), dtype=float)
            penalty = penalty_weight * float(np.sum(np.maximum(0.0, G) ** 2))
            F += penalty

        return F

    # ------------------------------------------------------------------
    # Copy for parallel workers
    # ------------------------------------------------------------------

    def make_copy(self) -> "CalibrationProblem":
        """Return a deep-copied :class:`CalibrationProblem` safe for parallel workers.

        The deep-copied model already holds its prepared state; data preparation
        is not re-run.  All mutable state (model, observed array, bounds) is
        independent from the original.

        :rtype: CalibrationProblem
        """
        cp = CalibrationProblem.__new__(CalibrationProblem)
        cp._model = copy.deepcopy(self._model)
        cp._observed = self._observed.copy()
        cp._objectives = list(self._objectives)
        cp._mask = None if self._mask is None else self._mask.copy()
        cp._objective_masks = [
            None if m is None else m.copy() for m in self._objective_masks
        ]
        cp._result_extractor = self._result_extractor
        cp._column_map = self._column_map
        cp._prepare_fn = self._prepare_fn
        cp._prepare_kwargs = self._prepare_kwargs
        cp._param_names = list(self._param_names)
        cp._xl = self._xl.copy()
        cp._xu = self._xu.copy()
        cp._n_ieq_constr = self._n_ieq_constr
        cp._constraint_names = list(self._constraint_names)
        cp._constraint_descriptions = list(self._constraint_descriptions)
        return cp
