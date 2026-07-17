"""EnsembleModel — generic multi-model compositor.

Combines any number of :class:`~sparsehydro.models.IModel` instances via a
**weighted sum** or **weighted product**.  All child model parameters are
re-registered under prefixed names so a single
:class:`~sparsehydro.calibration.CalibrationProblem` can calibrate the entire
ensemble end-to-end.  All child inequality constraints are surfaced and
concatenated so constraint-aware solvers (e.g. NSGA-II via pymoo) see them
natively.

Usage::

    from sparsehydro.models import EnsembleModel
    from sparsehydro.models.rdii import RDIIModel

    model_a = RDIIModel()
    model_b = RDIIModel()

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

from ..enums import ModelState
from .base import IModel
from ..parameters import ConstraintRecord, FieldRecord, ScalarParameter
from ..registry import registry


@registry.register
class EnsembleModel(IModel):
    """Combines N :class:`~sparsehydro.models.IModel` instances via weighted sum or product.

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
        Defaults to ``"model_1"``, ``"model_2"``, …
    :param output_name: Column name for the combined output in ``predict()`` output.
        Defaults to ``"ensemble_output"``.
    :param normalize_weights: If ``True`` (default) and ``mode="sum"``, a
        ``sum_w_leq_1`` constraint ``Σ w_i ≤ 1.0`` is registered.
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
        self._param_maps: list[dict[str, str]] = []
        self._param_owner: dict[str, str] = {}

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register weight + child parameters, surface child constraints."""
        n = len(self._children)

        for child in self._children:
            if child.is_created():
                child.initialize()
            if child.is_initialized():
                child.validate()

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

        self._param_owner = {}
        for i, alias in enumerate(self._aliases, 1):
            self._param_owner[f"w_{i}"] = "weights"
        for alias, mapping in zip(self._aliases, self._param_maps):
            for prefixed_name in mapping.values():
                self._param_owner[prefixed_name] = alias

        for child, alias in zip(self._children, self._aliases):
            for cname, cdesc in zip(
                child.inequality_constraint_names,
                child.inequality_constraint_descriptions,
            ):
                self.register_inequality_constraint(ConstraintRecord(
                    name=f"{alias}_{cname}",
                    description=f"[{alias}] {cdesc}",
                ))

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

        if self._normalize_weights:
            w_labels = " + ".join(f"w_{i}" for i in range(1, n + 1))
            self.register_inequality_constraint(ConstraintRecord(
                name="sum_w_leq_1",
                description=f"Σ mixing weights ({w_labels}) ≤ 1.0",
            ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Check all parameter bounds and advance to VALIDATED."""
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: Any, **kwargs: Any) -> None:
        """Sync ensemble parameters to children then call ``prepare(data)`` on each."""
        self._sync_to_children()
        for child in self._children:
            child.prepare(data, **kwargs)
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Sync parameters, run each child, and return the combined output."""
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
        """Concatenate child constraint residuals, then append ensemble-level constraints."""
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
        """Return a DataFrame summarising every registered scalar parameter."""
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
        """Update a parameter's value and/or bounds in-place."""
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
        """Apply every Pareto-front solution and return their combined outputs."""
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
        """Mapping of prefixed parameter name → component alias."""
        return dict(self._param_owner)
