# sparsehydro — Design Document

## Purpose

`sparsehydro` provides abstract interfaces and utilities for building *parsimonious*
(sparse-parameter) hydrological models in Python. The goal is to enforce a consistent
model lifecycle, typed parameter registries with bounds, and optional gradient-based
calibration via PyTorch.

---

## Package layout

```
sparsehydro/
├── .github/workflows/
│   ├── tests.yml        # pytest matrix: 3.10–3.12 × ubuntu/windows/macos
│   ├── docs.yml         # Sphinx → GitHub Pages
│   └── publish.yml      # PyPI publish on GitHub Release
├── docs/
│   ├── conf.py          # sphinx_rtd_theme + autodoc + napoleon
│   ├── index.rst
│   ├── api.rst
│   ├── getting_started.rst
│   └── Makefile
├── src/sparsehydro/
│   ├── __init__.py      # public API surface
│   ├── enums.py         # ModelState enum
│   ├── parameters.py    # ScalarParameter, VectorParameter
│   ├── interfaces.py    # IModel (ABC)
│   └── torch_model.py   # ITorchModel (IModel + nn.Module, optional)
├── tests/
│   ├── test_enums.py
│   ├── test_parameters.py
│   ├── test_interfaces.py
│   └── test_torch_model.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Model lifecycle

```
CREATED ──► INITIALIZED ──► VALIDATED ──► PREPARED ──► PREDICTED ──► FINALIZED
```

| State       | Entered by    | Description                                     |
|-------------|---------------|-------------------------------------------------|
| CREATED     | `__init__`    | Object allocated; no parameters registered.     |
| INITIALIZED | `initialize()`| Parameters registered; model structure set up.  |
| VALIDATED   | `validate()`  | All parameters within bounds; inputs verified.  |
| PREPARED    | `prepare()`   | Forcing / input data loaded and pre-processed.  |
| PREDICTED   | `predict()`   | Model run complete; outputs available.          |
| FINALIZED   | `finalize()`  | Resources released; object may be discarded.    |

---

## Parameter system

### ScalarParameter

```python
ScalarParameter(name, value, lower_bound, upper_bound, units="", description="")
```

- `is_valid() -> bool` — `lower_bound <= value <= upper_bound`
- `normalize() -> float` — maps value to [0, 1]
- `clamp() -> ScalarParameter` — returns copy clamped to bounds

### VectorParameter

```python
VectorParameter(name, values, lower_bounds, upper_bounds, units="", description="")
```

- Scalar bounds are broadcast to match the vector length.
- `size -> int` — number of elements
- `is_valid() -> bool` — element-wise bounds check
- `normalize() -> np.ndarray` — element-wise normalization to [0, 1]
- `clamp() -> VectorParameter` — element-wise clamp

---

## IModel interface

```python
class IModel(ABC):
    # Lifecycle
    def initialize(self) -> None: ...       # abstract
    def validate(self) -> bool: ...         # abstract
    def prepare(self, *args, **kwargs): ... # abstract
    def predict(self, *args, **kwargs): ... # abstract
    def finalize(self) -> None: ...         # abstract

    # State queries
    @property
    def state(self) -> ModelState: ...
    def is_created(self) -> bool: ...
    def is_initialized(self) -> bool: ...
    def is_validated(self) -> bool: ...
    def is_prepared(self) -> bool: ...
    def is_predicted(self) -> bool: ...
    def is_finalized(self) -> bool: ...

    # Parameter registry
    def register_scalar_parameter(self, param: ScalarParameter) -> None: ...
    def register_vector_parameter(self, param: VectorParameter) -> None: ...
    def get_scalar_parameter(self, name: str) -> ScalarParameter: ...
    def get_vector_parameter(self, name: str) -> VectorParameter: ...
    def parameters_valid(self) -> bool: ...
    @property
    def scalar_parameter_names(self) -> list[str]: ...
    @property
    def vector_parameter_names(self) -> list[str]: ...
```

---

## ITorchModel interface

Extends `IModel` *and* `torch.nn.Module` for differentiable models.
`forward()` is abstract; `predict()` delegates to `forward()` and advances state.
The entire class is guarded by `try/except ImportError` so `torch` remains optional.

```python
class ITorchModel(IModel, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor: ...

    def predict(self, *args, **kwargs) -> torch.Tensor:
        result = self.forward(*args, **kwargs)
        self._state = ModelState.PREDICTED
        return result
```

---

## Infrastructure

| Tool         | Purpose                                                  |
|--------------|----------------------------------------------------------|
| hatchling    | PEP 517/518 build backend (`pyproject.toml`)             |
| pytest       | Unit testing with `--cov` for coverage reporting         |
| codecov      | Coverage badge and PR annotations                        |
| Sphinx       | Documentation (`sphinx_rtd_theme`, autodoc, napoleon)    |
| GitHub Pages | Hosts the built HTML docs                                |
| PyPI trusted | Publish via OIDC (no stored API token needed)            |

---

## Optional extras

```
pip install sparsehydro            # core only (numpy)
pip install sparsehydro[torch]     # + PyTorch
pip install sparsehydro[docs]      # + Sphinx toolchain
pip install sparsehydro[dev]       # + pytest, coverage, docs
```
