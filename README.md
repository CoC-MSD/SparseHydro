# sparsehydro

**sparsehydro** provides abstract interfaces and utilities for building *parsimonious*
(sparse-parameter) hydrological models in Python, with a focus on stormwater systems.

[![Tests](https://github.com/cbuahin/sparsehydro/actions/workflows/tests.yml/badge.svg)](https://github.com/cbuahin/sparsehydro/actions/workflows/tests.yml)
[![Docs](https://github.com/cbuahin/sparsehydro/actions/workflows/docs.yml/badge.svg)](https://cbuahin.github.io/sparsehydro)
[![PyPI](https://img.shields.io/pypi/v/sparsehydro)](https://pypi.org/project/sparsehydro/)
[![Python](https://img.shields.io/pypi/pyversions/sparsehydro)](https://pypi.org/project/sparsehydro/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Model lifecycle** — enforced six-state progression:
  `CREATED → INITIALIZED → VALIDATED → PREPARED → PREDICTED → FINALIZED`
- **Parameter registry** — named scalar and vector parameters with lower/upper bounds,
  normalization, and clamping
- **PyTorch support** — `ITorchModel` combines the lifecycle interface with `nn.Module`
  for gradient-based calibration
- **pandas outputs** — `predict()` returns `pd.DataFrame | pd.Series`
- **Epytext docstrings** rendered via Sphinx + `sphinxcontrib-epytext`

## Installation

```bash
pip install sparsehydro
```

With PyTorch support:

```bash
pip install sparsehydro[torch]
```

## Quick start

```python
import pandas as pd
from sparsehydro import IModel, ModelState, ScalarParameter

class LinearReservoir(IModel):
    def initialize(self) -> None:
        self.register_scalar_parameter(
            ScalarParameter("k", value=0.3, lower_bound=0.0, upper_bound=1.0,
                            units="1/day", description="Recession coefficient")
        )
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, forcing: pd.Series) -> None:
        self._forcing = forcing
        self._state = ModelState.PREPARED

    def predict(self) -> pd.Series:
        k = self.get_scalar_parameter("k").value
        self._state = ModelState.PREDICTED
        return self._forcing * k

    def finalize(self) -> None:
        self._state = ModelState.FINALIZED


model = LinearReservoir()
model.initialize()
model.validate()
model.prepare(pd.Series([10.0, 8.0, 6.0], name="P_mm"))
print(model.predict())
model.finalize()
```

## Documentation

Full documentation is available at [cbuahin.github.io/sparsehydro](https://cbuahin.github.io/sparsehydro).

## License

MIT — see [LICENSE](LICENSE).
