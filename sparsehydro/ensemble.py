"""EnsembleModel — generic multi-model compositor.

Combines any number of :class:`~sparsehydro.interfaces.IModel` instances via a
**weighted sum** or **weighted product**.  All child model parameters are
re-registered under prefixed names so a single
:class:`~sparsehydro.calibration.CalibrationProblem` can calibrate the entire
ensemble end-to-end.  All child inequality constraints are surfaced and
concatenated so constraint-aware solvers (e.g. NSGA-II via pymoo) see them
natively.

Usage::

    from sparsehydro.ensemble import EnsembleModel
    from sparsehydro.rdii import RDIIModel

    model_a = RDIIModel(n_triangles=3)
    model_b = RDIIModel(n_triangles=2)

    ensemble = EnsembleModel(
        components=[
            (model_a, lambda df: df["rdii_cfs"].to_numpy()),
            (model_b, lambda df: df["rdii_cfs"].to_numpy()),
        ],
        mode="sum",
        aliases=["fast", "slow"],
        output_name="Q_total",
    )
    ensemble.initialize()
    ensemble.validate()
    ensemble.prepare(df)
    result = ensemble.predict()
    # DataFrame columns: datetime, fast_output, slow_output, Q_total

Combination formulas
--------------------
Let ``y_i[t]`` be the scalar output of child *i* and ``w_i`` its weight.

* **sum** mode:     ``output[t] = Σ_i  w_i · y_i[t]``
* **product** mode: ``output[t] = Π_i  y_i[t]^w_i``

Weight defaults: ``1 / N`` for sum; ``1.0`` for product.
Weight bounds: ``[0.0, 2.0]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from .enums import ModelState
from .interfaces import IModel
from .parameters import ConstraintRecord, FieldRecord, ScalarParameter
from .registry import registry


@registry.register
class EnsembleModel(IModel):
    """Combines N :class:`~sparsehydro.interfaces.IModel` instances via weighted sum or product.

    Parameters are namespaced as ``{alias}_{original_name}`` to prevent collisions.
    Mixing weights ``w_1 … w_N`` are calibratable
    :class:`~sparsehydro.parameters.ScalarParameter` objects exposed to the
    calibration framework.  All child inequality constraints are surfaced and
    concatenated so constraint-aware solvers handle them natively.

    :param components: List of ``(model, extractor)`` pairs.  ``extractor`` is a
        callable ``(predict_df: pandas.DataFrame) → numpy.ndarray`` that selects
        the scalar output array from each child's ``predict()`` result.
    :param mode: Combination mode.  ``"sum"`` computes ``Σ w_i · y_i``; ``"product"``
        computes ``Π y_i^w_i``.  Defaults to ``"sum"``.
    :param aliases: Optional label for each component used as a parameter prefix.
        Defaults to ``"model_1"``, ``"model_2"``, …  Must be the same length as
        *components* when provided.
    :param output_name: Column name for the combined output in ``predict()`` output.
        Defaults to ``"ensemble_output"``.
    :param normalize_weights: If ``True`` (default) and ``mode="sum"``, a
        ``sum_w_leq_1`` constraint ``Σ w_i ≤ 1.0`` is registered and enforced
        by the calibration framework.

    Example::

        ensemble = EnsembleModel(
            components=[
                (rdii_model,   lambda df: df["rdii_cfs"].to_numpy()),
                (custom_model, lambda df: df["Q"].to_numpy()),
            ],
            mode="sum",
            aliases=["rdii", "custom"],
            output_name="Q_total",
        )
        ensemble.initialize()
        ensemble.validate()
        ensemble.prepare(forcing_df)
        out = ensemble.predict()   # DataFrame: datetime, rdii_output, custom_output, Q_total
    """

    model_name = "ensemble"

    def __init__(
        self,
        components: list[tuple[IModel, Callable]],
        mode: str = "sum",
        aliases: list[str] | None = None,
        output_name: str = "ensemble_output",
        normalize_weights: bool = True,
    ) -> None:
        super().__init__()
        if not components:
            raise ValueError("EnsembleModel requires at least one component.")
        if mode not in ("sum", "product"):
            raise ValueError(f"mode must be 'sum' or 'product'; got {mode!r}.")

        n = len(components)
        if aliases is not None:
            if len(aliases) != n:
                raise ValueError(
                    f"aliases length ({len(aliases)}) must match components length ({n})."
                )
            self._aliases: list[str] = list(aliases)
        else:
            self._aliases = [f"model_{i + 1}" for i in range(n)]

        self._children: list[IModel] = [m for m, _ in components]
        self._extractors: list[Callable] = [e for _, e in components]
        self._mode: str = mode
        self._output_name: str = output_name
        self._normalize_weights: bool = normalize_weights and (mode == "sum")
        self._param_maps: list[dict[str, str]] = []  # {orig_name: prefixed_name} per child
        self._param_owner: dict[str, str] = {}  # prefixed_name → alias

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register weight + child parameters, surface child constraints.

        Any child still in CREATED or INITIALIZED state is automatically
        initialized/validated before its parameters are flattened.
        """
        n = len(self._children)

        # Auto-initialize and validate children that haven't been already
        for child in self._children:
            if child.is_created():
                child.initialize()
            if child.is_initialized():
                child.validate()

        # --- Weight parameters ---
        default_w = 1.0 / n if self._mode == "sum" else 1.0
        for i, alias in enumerate(self._aliases, start=1):
            self.register_scalar_parameter(ScalarParameter(
                name=f"w_{i}",
                value=default_w,
                lower_bound=0.0,
                upper_bound=2.0,
                units="-",
                description=f"Mixing weight for component '{alias}'",
            ))

        # --- Child parameter flattening with prefix ---
        self._param_maps = []
        for child, alias in zip(self._children, self._aliases):
            mapping: dict[str, str] = {}
            for orig_name in child.scalar_parameter_names:
                prefixed = f"{alias}_{orig_name}"
                p = child.get_scalar_parameter(orig_name)
                self.register_scalar_parameter(ScalarParameter(
                    name=prefixed,
                    value=p.value,
                    lower_bound=p.lower_bound,
                    upper_bound=p.upper_bound,
                    units=p.units,
                    description=f"[{alias}] {p.description}",
                    calibrate=p.calibrate,
                ))
                mapping[orig_name] = prefixed
            self._param_maps.append(mapping)

        # --- Build param_owner map ---
        self._param_owner = {}
        for i, alias in enumerate(self._aliases, 1):
            self._param_owner[f"w_{i}"] = "weights"
        for alias, mapping in zip(self._aliases, self._param_maps):
            for prefixed_name in mapping.values():
                self._param_owner[prefixed_name] = alias

        # --- Child constraint metadata (prefixed) ---
        for child, alias in zip(self._children, self._aliases):
            for cname, cdesc in zip(
                child.inequality_constraint_names,
                child.inequality_constraint_descriptions,
            ):
                self.register_inequality_constraint(ConstraintRecord(
                    name=f"{alias}_{cname}",
                    description=f"[{alias}] {cdesc}",
                ))

        # --- Output field metadata ---
        self.register_output_field(FieldRecord(
            name="datetime",
            description="Simulation time step",
        ))
        for child, alias in zip(self._children, self._aliases):
            child_calibratable = child.calibratable_output_names
            self.register_output_field(FieldRecord(
                name=f"{alias}_output",
                units="",
                description=(
                    f"Raw signal from component '{alias}' "
                    f"(calibratable outputs: {child_calibratable or 'n/a'})"
                ),
                calibratable=False,
            ))
        self.register_output_field(FieldRecord(
            name=self._output_name,
            units="",
            description=(
                f"Combined ensemble output ({self._mode} of "
                f"{len(self._aliases)} components: "
                f"{', '.join(self._aliases)})"
            ),
            calibratable=True,
        ))

        # --- Ensemble-level weight constraint ---
        if self._normalize_weights:
            w_labels = " + ".join(f"w_{i}" for i in range(1, n + 1))
            self.register_inequality_constraint(ConstraintRecord(
                name="sum_w_leq_1",
                description=f"Σ mixing weights ({w_labels}) ≤ 1.0",
            ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Check all parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds; ``False`` otherwise.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: Any, **kwargs: Any) -> None:
        """Sync ensemble parameters to children then call ``prepare(data)`` on each.

        :param data: Input data forwarded unchanged to every child model's ``prepare``.
        """
        self._sync_to_children()
        for child in self._children:
            child.prepare(data, **kwargs)
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Sync parameters, run each child, and return the combined output.

        :returns: DataFrame containing:

            * ``datetime`` — copied from the first child's output (when present).
            * ``{alias}_output`` — raw scalar output from each child (diagnostic).
            * ``{output_name}`` — combined ensemble signal.

        :rtype: pandas.DataFrame
        """
        self._sync_to_children()

        n = len(self._children)
        child_outputs: list[np.ndarray] = []
        first_pred: pd.DataFrame | None = None

        for i, (child, extractor) in enumerate(zip(self._children, self._extractors)):
            pred_df = child.predict()
            if i == 0:
                first_pred = pred_df
            child_outputs.append(np.asarray(extractor(pred_df), dtype=float))

        weights = [self.get_scalar_parameter(f"w_{i}").value for i in range(1, n + 1)]

        if self._mode == "sum":
            combined: np.ndarray = sum(  # type: ignore[assignment]
                w * y for w, y in zip(weights, child_outputs)
            )
        else:  # product
            combined = np.ones_like(child_outputs[0])
            for w, y in zip(weights, child_outputs):
                with np.errstate(invalid="ignore", divide="ignore"):
                    combined = combined * np.where(y > 0.0, y ** w, 0.0)

        out: dict[str, Any] = {}
        if first_pred is not None and "datetime" in first_pred.columns:
            out["datetime"] = first_pred["datetime"].to_numpy()
        for alias, y in zip(self._aliases, child_outputs):
            out[f"{alias}_output"] = y
        out[self._output_name] = combined

        self._state = ModelState.PREDICTED
        return pd.DataFrame(out)

    def finalize(self) -> None:
        """Finalize all child models and advance to FINALIZED."""
        for child in self._children:
            child.finalize()
        self._state = ModelState.FINALIZED

    def inequality_constraints(self) -> list[float]:
        """Concatenate child constraint residuals, then append ensemble-level constraints.

        Child residuals are returned in alias order.  For sum mode with
        ``normalize_weights=True``, ``Σ w_i − 1.0`` is appended last (feasible
        when ≤ 0).

        :returns: List of residuals ``g_j`` where ``g_j ≤ 0`` means feasible.
        :rtype: list[float]
        """
        g: list[float] = []
        for child in self._children:
            g.extend(child.inequality_constraints())
        if self._normalize_weights:
            n = len(self._children)
            w_sum = sum(
                self.get_scalar_parameter(f"w_{i}").value for i in range(1, n + 1)
            )
            g.append(w_sum - 1.0)
        return g

    # ------------------------------------------------------------------
    # Parameter inspection and configuration
    # ------------------------------------------------------------------

    def parameter_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising every registered scalar parameter.

        Columns: ``parameter``, ``group``, ``value``, ``lower_bound``,
        ``upper_bound``, ``units``, ``calibrate``, ``description``.

        The ``group`` column contains the component alias that owns the
        parameter (e.g. ``"rdii"``, ``"seas"``) or ``"weights"`` for
        the mixing-weight parameters.

        Intended for interactive inspection in Jupyter notebooks::

            display(ensemble.parameter_table().style.format({"value": "{:.4g}"}))

        :returns: One row per scalar parameter, in registration order.
        :rtype: pandas.DataFrame
        """
        rows = []
        for name in self.scalar_parameter_names:
            p = self.get_scalar_parameter(name)
            rows.append({
                "parameter":   name,
                "group":       self._param_owner.get(name, ""),
                "value":       p.value,
                "lower_bound": p.lower_bound,
                "upper_bound": p.upper_bound,
                "units":       p.units,
                "calibrate":   p.calibrate,
                "description": p.description,
            })
        return pd.DataFrame(rows)

    def set_parameter(
        self,
        name: str,
        *,
        value: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        calibrate: bool | None = None,
    ) -> None:
        """Update a parameter's value and/or bounds in-place.

        Convenience wrapper around :meth:`~sparsehydro.parameters.ScalarParameter.update`
        for ergonomic notebook configuration before calibration::

            ensemble.set_parameter("w_1", value=1.0, calibrate=False)
            ensemble.set_parameter("rdii_area_acres", lower_bound=100.0, upper_bound=2000.0)

        :param name: Parameter name as shown in :meth:`parameter_table`.
        :param value: New current value.
        :param lower_bound: New lower bound.
        :param upper_bound: New upper bound.
        :param calibrate: Whether the optimizer should adjust this parameter.
        :raises KeyError: If *name* is not registered.
        :raises ValueError: If the resulting bounds are invalid.
        """
        self.get_scalar_parameter(name).update(
            value=value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            calibrate=calibrate,
        )

    def collect_pareto_predictions(
        self,
        data: Any,
        result: Any,
        output_col: str | None = None,
        **prepare_kwargs: Any,
    ) -> np.ndarray:
        """Apply every Pareto-front solution and return their combined outputs.

        Iterates over ``result.pareto_X``, sets the corresponding
        parameter values, runs :meth:`prepare` and :meth:`predict`, and
        collects the specified output column.  Parameter values are saved
        before the loop and restored in a ``finally`` block so the model
        is left in its original state.

        .. note::
            After this call the model is in ``PREDICTED`` state.  Call
            ``ensemble.prepare(data)`` before the next ``ensemble.predict()``.

        :param data: Forcing data forwarded to :meth:`prepare`.
        :param result: A :class:`~sparsehydro.calibration.CalibrationResult`
            (or any object with ``.param_names`` and ``.pareto_X``).
        :param output_col: Column to extract from each ``predict()`` output.
            Defaults to :attr:`output_name`.
        :param prepare_kwargs: Extra keyword arguments forwarded to :meth:`prepare`.
        :returns: Array of shape ``(n_solutions, n_timesteps)``.
        :rtype: numpy.ndarray
        """
        if output_col is None:
            output_col = self._output_name

        saved = {
            name: self.get_scalar_parameter(name).value
            for name in result.param_names
        }

        preds: list[np.ndarray] = []
        try:
            for x in result.pareto_X:
                for name, value in zip(result.param_names, x):
                    self.get_scalar_parameter(name).value = float(value)
                self.prepare(data, **prepare_kwargs)
                sim = self.predict()
                preds.append(np.asarray(sim[output_col], dtype=float))
        finally:
            for name, value in saved.items():
                self.get_scalar_parameter(name).value = value

        return np.array(preds)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_to_children(self) -> None:
        """Push ensemble registry values back to each child model's registry."""
        for child, mapping in zip(self._children, self._param_maps):
            for orig_name, prefixed_name in mapping.items():
                child.get_scalar_parameter(orig_name).value = (
                    self.get_scalar_parameter(prefixed_name).value
                )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Combination mode: ``"sum"`` or ``"product"``."""
        return self._mode

    @property
    def aliases(self) -> list[str]:
        """Alias label for each component (used as parameter prefix)."""
        return list(self._aliases)

    @property
    def output_name(self) -> str:
        """Column name for the combined output in ``predict()`` results."""
        return self._output_name

    @property
    def n_components(self) -> int:
        """Number of child models."""
        return len(self._children)

    @property
    def param_owner(self) -> dict[str, str]:
        """Mapping of prefixed parameter name → component alias.

        Keys are parameter names as registered (e.g. ``"rdii_ia_max"``).
        Values are the owning alias (e.g. ``"rdii"``) or ``"weights"``
        for the mixing-weight parameters.  Available after :meth:`initialize`.

        :rtype: dict[str, str]
        """
        return dict(self._param_owner)
