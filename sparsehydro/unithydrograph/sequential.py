"""Sequential event-by-event unit hydrograph fitting.

Uses :class:`~sparsehydro.calibration.CalibrationProblem` for data preparation
and objective evaluation, and constructs a
:class:`~sparsehydro.calibration.CalibrationResult` per event.

Warm starting from the previous event's fitted parameters is handled by calling
``scipy.optimize.minimize`` directly (bypassing ``ScipySolver``, which always
restarts from the bounds midpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize as scipy_minimize
from scipy.optimize import differential_evolution
from scipy.signal import savgol_filter

from ..calibration.objectives import RMSE, KGE, NashSutcliffe, IObjective
from ..calibration.problem import CalibrationProblem
from ..calibration.result import CalibrationResult
from ..interfaces import IModel
from ..events import EventRecord


@dataclass
class SequentialFitSummary:
    """Results of a sequential unit hydrograph fitting run.

    Attributes
    ----------
    events : list[EventRecord]
        Events that were fitted, in chronological order.
    calibration_results : list[CalibrationResult]
        One :class:`~sparsehydro.calibration.CalibrationResult` per event.
        ``pareto_X[0]`` holds fitted parameters;
        ``objective_display_values()[0]`` returns [RMSE, NSE, KGE].
    global_predicted : pd.DataFrame
        Full-domain accumulated predicted flow.  Columns: ``datetime``, ``Q_pred``.
    global_observed : pd.DataFrame
        Full-domain stormflow.  Columns: ``datetime``, ``stormflow``.
    model_class_name : str
        Name of the model class produced by ``model_factory``.
    """

    events: list[EventRecord]
    calibration_results: list[CalibrationResult]
    global_predicted: pd.DataFrame
    global_observed: pd.DataFrame
    model_class_name: str

    def parameter_evolution(self) -> pd.DataFrame:
        """Return fitted parameters indexed by event.

        Columns: ``event_id``, ``start_datetime``, ``end_datetime``,
        then one column per calibrated parameter.
        """
        rows = []
        for event, result in zip(self.events, self.calibration_results):
            row: dict = {
                "event_id": event.event_id,
                "start_datetime": event.start_datetime,
                "end_datetime": event.end_datetime,
            }
            row.update(dict(zip(result.param_names, result.pareto_X[0])))
            rows.append(row)
        return pd.DataFrame(rows)

    def metrics_summary(self) -> pd.DataFrame:
        """Return per-event objective values in display form.

        Columns: ``event_id``, then one column per objective name.
        """
        rows = []
        for event, result in zip(self.events, self.calibration_results):
            display_F = result.objective_display_values()[0]
            row: dict = {"event_id": event.event_id}
            for name, val in zip(result.objective_names, display_F):
                row[name] = val
            rows.append(row)
        return pd.DataFrame(rows)


class SequentialFitter:
    """Fit an :class:`~sparsehydro.interfaces.IModel` to storm events one by one.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a fresh :class:`~sparsehydro.interfaces.IModel`
        in CREATED state.  Works with single UH models and
        :class:`~sparsehydro.ensemble.EnsembleModel`::

            SequentialFitter(lambda: GammaUH(), data, events)
            SequentialFitter(lambda: make_ensemble(), data, events)

    rain_stormflow : pd.DataFrame
        Full time series with columns ``datetime``, ``rain``, ``stormflow``.
    events : list[EventRecord]
        Events to fit (sorted internally by start_datetime).
    output_column : str
        Column name produced by ``model.predict()``.  Default ``"Q_pred"``.
    """

    def __init__(
        self,
        model_factory: Callable[[], IModel],
        rain_stormflow: pd.DataFrame,
        events: list[EventRecord],
        output_column: str = "Q_pred",
    ) -> None:
        self._factory = model_factory
        self._data = rain_stormflow.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._data["stormflow"] = np.maximum(self._data["stormflow"].values, 0.0)
        self._events = sorted(events, key=lambda e: e.start_datetime)
        self._output_column = output_column

    def fit(
        self,
        time_range: tuple[str, str] | None = None,
        objectives: list[IObjective] | None = None,
        calibration_objective_index: int = 0,
        method: str = "Nelder-Mead",
        smooth_obs: bool = True,
        verbose: bool = True,
    ) -> SequentialFitSummary:
        """Run the sequential fitting loop.

        Parameters
        ----------
        time_range : (start, end) str tuple, optional
            Restrict fitting to this date range.
        objectives : list[IObjective], optional
            Defaults to ``[RMSE(), NashSutcliffe(), KGE()]``.
        calibration_objective_index : int
            Index of *objectives* to minimise (default 0 = RMSE).
        method : str
            ``scipy.optimize.minimize`` method, or ``"differential_evolution"``
            for a global search.
        smooth_obs : bool
            Light Savitzky-Golay smooth on observed flow within each event.
        verbose : bool
        """
        if objectives is None:
            objectives = [RMSE(), NashSutcliffe(), KGE()]

        data = self._filter_data(time_range)
        events = self._filter_events(time_range)

        if not events:
            return SequentialFitSummary(
                events=[],
                calibration_results=[],
                global_predicted=pd.DataFrame({"datetime": data["datetime"], "Q_pred": np.zeros(len(data))}),
                global_observed=data[["datetime", "stormflow"]].copy(),
                model_class_name=self._factory().__class__.__name__,
            )

        datetime_arr = data["datetime"].values
        Q_residual = data["stormflow"].values.copy()
        Q_global = np.zeros(len(data))

        cal_results: list[CalibrationResult] = []
        fitted_events: list[EventRecord] = []
        last_x: np.ndarray | None = None
        last_param_names: list[str] = []

        n_events = len(events)
        end_dt = pd.Timestamp(datetime_arr[-1])

        for i, event in enumerate(events):
            if verbose:
                print(f"[{i+1}/{n_events}] Event {event.event_id}: {event.start_datetime} → {event.end_datetime}")

            fit_mask = (data["datetime"] >= event.start_datetime) & (data["datetime"] <= event.end_datetime)
            if not fit_mask.any():
                if verbose:
                    print("  Skipped: no data in event window.")
                continue

            fit_indices = np.where(fit_mask.values)[0]
            idx_s, idx_e = int(fit_indices[0]), int(fit_indices[-1]) + 1

            app_end = events[i + 1].start_datetime if i < n_events - 1 else end_dt
            app_mask = (data["datetime"] >= event.start_datetime) & (data["datetime"] < app_end)
            app_indices = np.where(app_mask.values)[0]

            event_data = data[fit_mask].copy().reset_index(drop=True)
            obs = Q_residual[idx_s:idx_e].copy()
            if smooth_obs and len(obs) >= 5:
                wlen = 5
                obs = savgol_filter(obs, window_length=wlen, polyorder=2, mode="interp")
            obs = np.maximum(obs, 0.0)
            event_data["stormflow"] = obs

            model_i = self._factory()
            model_i.initialize()
            if last_x is not None and last_param_names:
                for name, val in zip(last_param_names, last_x):
                    if name in model_i.scalar_parameter_names:
                        p = model_i.get_scalar_parameter(name)
                        p.update(value=float(np.clip(val, p.lower_bound, p.upper_bound)))
            model_i.validate()

            try:
                problem = CalibrationProblem(
                    model=model_i,
                    data=event_data,
                    objectives=objectives,
                    column_map={"observed": "stormflow", "predicted": self._output_column},
                )
            except Exception as exc:
                if verbose:
                    print(f"  CalibrationProblem failed: {exc!r} — skipped.")
                continue

            xl, xu = problem.bounds
            x0 = np.clip(last_x, xl, xu) if (last_x is not None and last_param_names == problem.param_names) else (xl + xu) / 2.0

            best_x, success = self._optimise(problem, x0, xl, xu, method, calibration_objective_index)

            all_F = problem.evaluate(best_x).reshape(1, -1)
            cal_result = CalibrationResult(
                history=[],
                pareto_X=best_x.reshape(1, -1),
                pareto_F=all_F,
                param_names=problem.param_names,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )

            if verbose:
                disp = cal_result.objective_display_values()[0]
                obj_strs = ", ".join(f"{n}={v:.4f}" for n, v in zip(cal_result.objective_names, disp))
                print(f"  {obj_strs}, success={success}")

            for name, val in zip(problem.param_names, best_x):
                if name in model_i.scalar_parameter_names:
                    model_i.get_scalar_parameter(name).update(value=float(val))

            app_data = data[app_mask].copy().reset_index(drop=True)
            if len(app_data) > 0:
                try:
                    model_i.prepare(app_data)
                    app_pred = model_i.predict()
                    q_app = app_pred[self._output_column].values
                    pred_len = min(len(q_app), len(app_indices))
                    Q_global[app_indices[:pred_len]] += q_app[:pred_len]
                    Q_residual[app_indices[:pred_len]] -= q_app[:pred_len]
                    Q_residual = np.maximum(Q_residual, 0.0)
                except Exception as exc:
                    if verbose:
                        print(f"  Application window prediction failed: {exc!r}")

            last_x = best_x
            last_param_names = problem.param_names
            cal_results.append(cal_result)
            fitted_events.append(event)

        return SequentialFitSummary(
            events=fitted_events,
            calibration_results=cal_results,
            global_predicted=pd.DataFrame({"datetime": datetime_arr, "Q_pred": Q_global}),
            global_observed=data[["datetime", "stormflow"]].copy(),
            model_class_name=self._factory().__class__.__name__,
        )

    def _optimise(self, problem, x0, xl, xu, method, obj_idx):
        bounds_list = list(zip(xl.tolist(), xu.tolist()))

        def scalar_fn(x):
            return float(problem.evaluate(x)[obj_idx])

        if method == "differential_evolution":
            res = differential_evolution(scalar_fn, bounds=bounds_list, seed=42, polish=True)
            return res.x, bool(res.success)

        try:
            res = scipy_minimize(
                scalar_fn, x0=x0, bounds=Bounds(lb=xl, ub=xu), method=method,
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
            )
            return res.x, bool(res.success)
        except Exception:
            return x0.copy(), False

    def _filter_data(self, time_range):
        data = self._data.copy()
        if time_range is not None:
            s, e = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
            data = data[(data["datetime"] >= s) & (data["datetime"] <= e)]
        return data.reset_index(drop=True)

    def _filter_events(self, time_range):
        evts = self._events
        if time_range is not None:
            s, e = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
            evts = [ev for ev in evts if ev.start_datetime >= s and ev.end_datetime <= e]
        return evts


__all__ = ["SequentialFitter", "SequentialFitSummary"]
