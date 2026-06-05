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

from typing import Any, Callable

import numpy as np
import pandas as pd

from .enums import ModelState
from .interfaces import IModel
from .parameters import ConstraintRecord, ScalarParameter
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
