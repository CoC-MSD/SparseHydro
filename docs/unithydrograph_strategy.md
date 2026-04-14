# Unit Hydrograph Module — Implementation Strategy

## Goal

Implement the unit hydrograph (UH) rainfall-runoff transformation as a
first-class module of `sparsehydro`, using the `IModel` lifecycle and parameter
registry, extensible to future UH types.

---

## Class hierarchy

```
IModel  (sparsehydro.interfaces)
└── IUnitHydrograph  (sparsehydro.unithydrograph.base)
    └── NashUnitHydrograph  (sparsehydro.unithydrograph.nash)
    └── [future: ClarkUH, SnyderUH, GIUH, …]
```

---

## Module layout

```
src/sparsehydro/unithydrograph/
├── __init__.py          # exports IUnitHydrograph, NashUnitHydrograph
├── base.py              # IUnitHydrograph abstract class
└── nash.py              # NashUnitHydrograph + @registry.register
```

A **subpackage** (not a single file) is used from the start because:

- New UH types (Clark, Snyder, GIUH) slot in as peer modules without
  touching existing code.
- Each model file is independently importable and testable.
- Sphinx autodoc can document each module separately.

---

## `IUnitHydrograph` design

### What it adds to `IModel`

| Addition | Kind | Purpose |
|---|---|---|
| `unit_response(times)` | **abstract** | Evaluate IUH ordinates (1/hr) at given times (hr) |
| `validate()` | concrete | Checks `parameters_valid()`; advances state |
| `prepare(excess_rainfall, dt_hours)` | concrete | Stores input; infers Δt from index |
| `predict()` | concrete | Calls `_convolve()`; advances state |
| `finalize()` | concrete | Releases stored data |
| `_convolve(excess, dt)` | concrete (private) | Discrete linear convolution |

### Subclass contract

A concrete UH subclass only needs:

```python
class MyUH(IUnitHydrograph):
    model_name = "my-uh"

    def initialize(self) -> None:
        self.register_scalar_parameter(...)
        self._state = ModelState.INITIALIZED

    def unit_response(self, times: np.ndarray) -> np.ndarray:
        # return gamma/empirical/tabulated ordinates
        ...
```

`validate`, `prepare`, `predict`, and `finalize` are **inherited** from
`IUnitHydrograph` and do not need to be re-implemented.

### Convolution algorithm

```
1. Evaluate UH ordinates over a window of 3 × storm duration.
2. Truncate where cumulative mass reaches 99.9 % of total.
3. Multiply by Δt → dimensionless discrete ordinates (sum ≈ 1).
4. np.convolve(excess, uh, mode="full")[:N]
```

Truncation at 99.9 % keeps convolution fast for long time series while
preserving volume to within 0.1 %.

### Time-step inference (in `prepare`)

| Index type | Inference rule |
|---|---|
| `DatetimeIndex` | `(index[1] - index[0]).total_seconds() / 3600` |
| Numeric | `index[1] - index[0]` (interpreted as hours) |
| Single-element | Raises `ValueError`; pass `dt_hours` explicitly |

---

## `NashUnitHydrograph` design

### Parameters

| Name | Default | Bounds | Units | Meaning |
|---|---|---|---|---|
| `n` | 3.0 | [1, 20] | — | Number of linear reservoirs (shape) |
| `k` | 2.0 | [0.1, 100] | hr | Reservoir storage coefficient (scale) |

### IUH formula

```
u(t; n, k) = (1 / (k · Γ(n))) · (t/k)^(n−1) · exp(−t/k),   t > 0
```

Statistical properties:
- Mean lag (centroid): **n · k**
- Mode (time to peak): **(n − 1) · k**  (for n > 1)
- Variance: **n · k²**

### Registration

`NashUnitHydrograph` is decorated with `@registry.register` so it is
available globally as soon as `sparsehydro` is imported:

```python
from sparsehydro.registry import registry
model = registry.create("nash-uh")
```

---

## Test strategy

| Test class | What is verified |
|---|---|
| `TestRegistration` | `model_name`, presence in global registry, `isinstance` check |
| `TestLifecycle` | Full `CREATED → … → FINALIZED` state progression |
| `TestTimestepInference` | Explicit dt, `DatetimeIndex` (hourly, 30 min), numeric index, single-element error |
| `TestUnitResponse` | Non-negativity, ∫u dt = 1, correct mode and centroid |
| `TestConvolution` | Volume conservation (≤ 1 % error), index preservation, zero-rain → zero-runoff, peak lag causality |

---

## Adding a new UH type

1. Create `src/sparsehydro/unithydrograph/my_uh.py`.
2. Subclass `IUnitHydrograph`, define `model_name`, implement `initialize`
   and `unit_response`.
3. Add `@registry.register` to auto-register on import.
4. Import the new class in `src/sparsehydro/unithydrograph/__init__.py`.
5. Add a `tests/test_my_uh.py` covering the same test categories above.

No changes to `IUnitHydrograph`, `ModelRegistry`, or `IModel` are required.

---

## Future extensions

| Extension | Approach |
|---|---|
| Differentiable Nash UH | Subclass `ITorchModel` + `IUnitHydrograph` (multiple inheritance); use `torch.special.gammaln` |
| Tabulated (SCS dimensionless) UH | Subclass `IUnitHydrograph`; `unit_response` interpolates from tabulated ordinates |
| Clark UH | Add `VectorParameter` for the time-area diagram; override `_convolve` to apply linear reservoir routing after translation |
| GIUH | Derive `unit_response` from geomorphological parameters (Horton ratios) |
