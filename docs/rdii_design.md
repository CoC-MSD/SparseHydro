# RDII Physics-Based Module — Design Document

## Motivation

SWMM's standard RDII calibration requires 12 monthly RTK parameter sets because a single set conflates soil moisture (Initial Abstraction) state with infrastructure leakage fraction. This module replaces monthly RTK tables with a physics-based Initial Abstraction (IA) recovery model driven by temperature and elapsed dry time, producing emergent seasonal RDII variation from one parameter set per sewershed.

---

## Module Layout

```
sparsehydro/rdii/
├── __init__.py               # Public surface; guards optimization/viz imports
├── initial_abstraction.py    # IAModel class + static compute_excess_series()
├── rtk_triangle.py           # RTKTriangle dataclass + triangular_uh()
├── model.py                  # RDIIModel(IModel), model_name="rdii"
├── objectives.py             # peak_weighted_mse(), nash_sutcliffe()
├── optimization.py           # RDIIOptimizer, RDIIResult, GenerationRecord
└── visualization.py          # plot_timeseries(), plot_pareto_evolution(),
                              # plot_parallel_coordinates()
```

Install optional dependencies with:

```bash
pip install sparsehydro[rdii]   # pymoo>=0.6, plotly>=5.0, scipy>=1.10
```

---

## Mathematics

### Initial Abstraction Recovery (dry interval)

During dry periods, available capacity recovers toward `IA_max` with an
exponential approach driven by a temperature-dependent rate:

$$IA_{avail}(t+\Delta t) = IA_{max} - \bigl(IA_{max} - IA_{avail}(t)\bigr)
  \cdot e^{-k_{rec}(T)\,\Delta t}$$

$$k_{rec}(T) = \begin{cases}
  k_0 + k_T \cdot e^{\,\theta\,(T - T_{ref})} & T \geq T_{freeze} \\
  0 & T < T_{freeze}
\end{cases}$$

- `k_0` — base rate capturing gravity drainage and capillary redistribution.
- `k_T · exp(θ(T − T_ref))` — thermally-driven evapotranspiration.
- Recovery is suppressed below `T_freeze`, producing high winter RDII naturally.

### Initial Abstraction Depletion (storm pulse)

During rainfall, capacity is depleted exponentially per mm of rainfall:

$$IA_{avail}(t+\Delta t) = IA_{avail}(t) \cdot e^{-k_{dep}\,\Delta P}$$

Rainfall excess passed to the RTK triangles:

$$P_{excess}(t) = \max\bigl(0,\; P(t) - IA_{avail}(t)\bigr)$$

### Triangular RTK Unit Hydrograph

For each triangle *i* with parameters $(R_i, T_i, K_i)$:

- **Time base:** $T_i \cdot (1 + K_i)$
- **Peak ordinate:** $\dfrac{2}{T_i \cdot (1 + K_i)}$ (ensures unit area)
- **RDII contribution:** $R_i \cdot \bigl(P_{excess} \circledast h_i\bigr)$

Total RDII flow:

$$Q_{RDII}(t) = \sum_{i=1}^{N} R_i \cdot \bigl(P_{excess} \circledast h_i\bigr)(t)$$

The convolution uses `numpy.convolve` for short series and
`scipy.signal.fftconvolve` when `max(n, kernel_len) > 500`.

---

## Parameter Specification

### Initial Abstraction — 7 parameters

| Name          | Default | Bounds        | Units | Description                              |
|---------------|---------|---------------|-------|------------------------------------------|
| `ia_max`      | 5.0     | [0.1, 50.0]   | mm    | Maximum abstraction capacity             |
| `ia_k0`       | 0.05    | [0.001, 1.0]  | 1/hr  | Base recovery rate (gravity drainage)    |
| `ia_kT`       | 0.02    | [0.0, 0.5]    | 1/hr  | Temperature-sensitive recovery coeff.   |
| `ia_theta`    | 0.1     | [0.0, 0.5]    | 1/°C  | Temperature sensitivity exponent         |
| `ia_T_ref`    | 20.0    | [0.0, 30.0]   | °C    | Reference temperature for ET scaling     |
| `ia_k_dep`    | 0.3     | [0.01, 5.0]   | 1/mm  | Depletion rate per mm of rainfall        |
| `ia_T_freeze` | 0.0     | [-5.0, 5.0]   | °C    | Recovery suppressed below this temp.    |

### RTK Triangle — 3 parameters per triangle i (1..N)

| Name  | Default (i=1/2/3)   | Bounds       | Units | Description                          |
|-------|---------------------|--------------|-------|--------------------------------------|
| `R_i` | 0.05 / 0.03 / 0.02  | [0.0, 1.0]   | —     | Fraction of P_excess entering sewer  |
| `T_i` | 1.0 / 12.0 / 72.0   | [0.1, 240.0] | hr    | Time to peak                         |
| `K_i` | 1.5 / 2.0 / 3.0     | [1.001, 10.0]| —     | Recession-to-peak ratio (strictly >1)|

**Constraint:** `sum(R_i) ≤ 1.0` — enforced in `RDIIModel.validate()`.

For N=3: 7 + 9 = **16 ScalarParameters** total.

---

## `RDIIModel` Usage

```python
import pandas as pd
from sparsehydro.rdii import RDIIModel

# Build input DataFrame
df = pd.DataFrame({
    "datetime":      pd.date_range("2024-01-01", periods=72, freq="h"),
    "rainfall_mm":   rainfall_array,    # required
    "flow_cfs":      observed_flow,     # optional (used by optimizer)
    "temperature_c": temperature_array, # optional (defaults to ia_T_ref)
})

model = RDIIModel(n_triangles=3)
model.initialize()
model.validate()
model.prepare(df)
result = model.predict()
# result columns: datetime, rdii_mm, p_excess_mm
model.finalize()
```

`prepare()` infers `dt_hours` from the median inter-row timedelta.
Missing `temperature_c` is filled with the current `ia_T_ref` parameter value.

---

## Multi-Objective Optimization

`RDIIOptimizer` wraps NSGA-II from `pymoo` and auto-discovers all
`ScalarParameter` bounds from the model registry.

### Objectives

1. **Peak-weighted MSE** (minimize):

$$PWMSE = \frac{\sum_t w_t\,(Q_{obs,t} - Q_{pred,t})^2}{\sum_t w_t},
\quad w_t = \frac{Q_{obs,t}}{\overline{Q}_{obs}}$$

2. **Nash-Sutcliffe Efficiency** (maximize → minimize −NSE):

$$NSE = 1 - \frac{\sum_t (Q_{obs,t}-Q_{pred,t})^2}{\sum_t (Q_{obs,t}-\overline{Q}_{obs})^2}$$

### Usage

```python
from sparsehydro.rdii import RDIIOptimizer

opt = RDIIOptimizer(model, observed_flow=df["flow_cfs"].to_numpy())
result = opt.run(pop_size=100, n_gen=200, seed=42)

print(result.pareto_nse)          # NSE values on final Pareto front
best_params = result.best_by_nse() # parameter vector with highest NSE
```

**NSGA-II settings:** SBX crossover (prob=0.9, η=15), polynomial mutation (η=20),
random float sampling, population stored every generation.

---

## Visualization

### Time Series

```python
from sparsehydro.rdii import plot_timeseries

fig = plot_timeseries(
    datetime=df["datetime"],
    rainfall_mm=df["rainfall_mm"],
    observed_flow=df["flow_cfs"],
    predicted_flow=result_df["rdii_mm"],
)
fig.show()
```

Dual-axis layout: rainfall bar chart (top row, Y-axis inverted) and
observed vs predicted flow lines (bottom row), shared X-axis.

### Pareto Front Evolution

```python
from sparsehydro.rdii import plot_pareto_evolution

fig = plot_pareto_evolution(opt_result)
fig.show()
```

Animated scatter plot with a generation slider. Each frame shows all
population solutions (grey) with Pareto-front solutions highlighted
(Viridis colorscale by NSE). A Play button animates the evolution.

### Parallel Coordinates

```python
from sparsehydro.rdii import plot_parallel_coordinates

fig = plot_parallel_coordinates(opt_result, color_by="nse")
fig.show()
```

One line per Pareto solution; axes = all calibrated parameters + PWMSE + NSE.
Drag the endpoints of any axis to filter solutions (Plotly `constraintrange`).
Color encodes NSE (or PWMSE when `color_by="pwmse"`).

---

## `RDIIResult` API

| Attribute / Method       | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| `history`                | `list[GenerationRecord]` — one per generation                 |
| `pareto_X`               | `ndarray (m, n_params)` — Pareto parameter vectors            |
| `pareto_F`               | `ndarray (m, 2)` — [PWMSE, −NSE] for Pareto solutions         |
| `pareto_nse`             | `ndarray (m,)` — NSE values (−pareto_F[:, 1])                 |
| `pareto_pwmse`           | `ndarray (m,)` — PWMSE values                                 |
| `param_names`            | `list[str]` — parameter names in column order                 |
| `best_by_nse()`          | Parameter vector with highest NSE                             |
| `best_by_pwmse()`        | Parameter vector with lowest PWMSE                            |
| `to_pareto_dataframe()`  | Tidy DataFrame: generation, params, pwmse, nse, is_pareto     |

---

## Verification Checklist

1. `pytest tests/test_rdii_model.py -v` — all tests pass.
2. `np.sum(triangular_uh(tri, dt)) * dt ≈ 1.0` for multiple (T, K, dt) combos.
3. Zero-rainfall input → zero RDII output.
4. `registry.is_registered("rdii")` → `True` after `import sparsehydro`.
5. NSE of best Pareto solution improves between generation 1 and generation N.
6. Plotly figures render: dual-axis time series, slider navigates generations,
   parallel coord axis dragging filters solutions correctly.
