"""CalibrationProblem: wraps any IModel with observed data and objectives.

Usage::

    from sparsehydro.calibration import CalibrationProblem, NashSutcliffe, PeakWeightedMSE

    problem = CalibrationProblem(
        model=prepared_model,
        observed=observed_flow,
        objectives=[PeakWeightedMSE(), NashSutcliffe()],
        result_extractor=lambda df: df["rdii_mm"].to_numpy(),
    )
    F = problem.evaluate(x)   # returns objective values in minimisation form
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from .objectives import IObjective

if TYPE_CHECKING:
    from ..interfaces import IModel


class CalibrationProblem:
    """General-purpose calibration problem for any :class:`~sparsehydro.interfaces.IModel`.

    Bundles a prepared model with observed data, a list of objectives, and a
    callable that extracts the predicted time series from the model's
    ``predict()`` output.

    Objective values returned by :meth:`evaluate` are in **minimisation form**:
    maximised objectives (``minimize=False``) are negated so that solvers can
    always minimise every column of the *F* matrix.

    :param model: A prepared model (``model.is_prepared() == True``).
    :type model: IModel
    :param observed: 1-D array of observed values aligned with the model time
        series.
    :type observed: numpy.ndarray
    :param objectives: At least one :class:`~sparsehydro.calibration.objectives.IObjective`.
    :type objectives: list[IObjective]
    :param result_extractor: Callable ``(pd.DataFrame) -> np.ndarray`` that
        extracts the predicted array from the DataFrame returned by
        ``model.predict()``.
        Example: ``lambda df: df["rdii_mm"].to_numpy()``
    :type result_extractor: Callable
    :raises RuntimeError: If ``model.is_prepared()`` is ``False``.
    :raises ValueError: If ``objectives`` is empty.
    """

    def __init__(
        self,
        model: "IModel",
        observed: np.ndarray,
        objectives: list[IObjective],
        result_extractor: Callable[[pd.DataFrame], np.ndarray],
    ) -> None:
        if not model.is_prepared():
            raise RuntimeError(
                "model must be in PREPARED state. Call model.prepare(data) first."
            )
        if not objectives:
            raise ValueError("At least one objective is required.")

        self._model = model
        self._observed = np.asarray(observed, dtype=float)
        self._objectives = list(objectives)
        self._result_extractor = result_extractor

        self._param_names = list(model.scalar_parameter_names)
        self._xl = np.array(
            [model.get_scalar_parameter(n).lower_bound for n in self._param_names],
            dtype=float,
        )
        self._xu = np.array(
            [model.get_scalar_parameter(n).upper_bound for n in self._param_names],
            dtype=float,
        )

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

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Apply a parameter vector, run ``predict()``, and return objectives.

        Objective values are returned in **minimisation form**: maximised
        objectives are negated.

        :param x: Parameter vector of length :attr:`n_params`.
        :type x: numpy.ndarray
        :returns: 1-D array of length :attr:`n_objectives` in minimisation form.
        :rtype: numpy.ndarray
        """
        for name, val in zip(self._param_names, x):
            self._model.get_scalar_parameter(name).value = float(val)

        pred_df = self._model.predict()
        predicted = self._result_extractor(pred_df)

        F = np.empty(len(self._objectives), dtype=float)
        for i, obj in enumerate(self._objectives):
            try:
                val = obj.evaluate(self._observed, predicted)
            except Exception:
                val = 1e12 if obj.minimize else -1e12
            F[i] = val if obj.minimize else -val
        return F

    # ------------------------------------------------------------------
    # Copy for parallel workers
    # ------------------------------------------------------------------

    def make_copy(self) -> "CalibrationProblem":
        """Return a deep-copied :class:`CalibrationProblem` safe for parallel workers.

        The copy has its own independent model and data, so parameter values
        can be mutated freely without affecting the original.

        :rtype: CalibrationProblem
        """
        return CalibrationProblem(
            model=copy.deepcopy(self._model),
            observed=self._observed.copy(),
            objectives=list(self._objectives),
            result_extractor=self._result_extractor,
        )
