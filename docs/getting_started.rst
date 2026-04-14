Getting Started
===============

Installation
------------

Install the core package (requires Python 3.10+):

.. code-block:: bash

   pip install sparsehydro

To enable gradient-based model calibration with PyTorch:

.. code-block:: bash

   pip install sparsehydro[torch]

Quick Example
-------------

Implement a concrete model by subclassing :class:`~sparsehydro.interfaces.IModel`:

.. code-block:: python

   import pandas as pd
   from sparsehydro import IModel, ModelState, ScalarParameter, VectorParameter

   class LinearReservoir(IModel):

       def initialize(self) -> None:
           self.register_scalar_parameter(
               ScalarParameter("k", value=0.3,
                               lower_bound=0.0, upper_bound=1.0,
                               units="1/day",
                               description="Recession coefficient")
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
           result = self._forcing * k
           self._state = ModelState.PREDICTED
           return result

       def finalize(self) -> None:
           self._state = ModelState.FINALIZED


   # Run the model
   import pandas as pd

   model = LinearReservoir()
   model.initialize()
   model.validate()
   model.prepare(forcing=pd.Series([10.0, 8.0, 6.0], name="rainfall_mm"))
   output = model.predict()
   model.finalize()
   print(output)

Differentiable Models with PyTorch
-----------------------------------

For gradient-based parameter estimation, inherit from
:class:`~sparsehydro.torch_model.ITorchModel`:

.. code-block:: python

   import torch
   import torch.nn as nn
   from sparsehydro import ModelState, ScalarParameter
   from sparsehydro.torch_model import ITorchModel

   class DiffReservoir(ITorchModel):

       def initialize(self) -> None:
           self.k = nn.Parameter(torch.tensor(0.3))
           self.register_scalar_parameter(
               ScalarParameter("k", value=0.3, lower_bound=0.0, upper_bound=1.0)
           )
           self._state = ModelState.INITIALIZED

       def validate(self) -> bool:
           ok = self.parameters_valid()
           if ok:
               self._state = ModelState.VALIDATED
           return ok

       def prepare(self, forcing: torch.Tensor) -> None:
           self._forcing = forcing
           self._state = ModelState.PREPARED

       def forward(self, forcing: torch.Tensor) -> torch.Tensor:
           return self.k * forcing

       def finalize(self) -> None:
           self._state = ModelState.FINALIZED

   model = DiffReservoir()
   model.initialize()
   model.validate()
   forcing = torch.tensor([10.0, 8.0, 6.0])
   model.prepare(forcing)

   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
   target = torch.tensor([3.0, 2.4, 1.8])

   for _ in range(100):
       optimizer.zero_grad()
       pred = model.predict(forcing)
       loss = ((pred - target) ** 2).mean()
       loss.backward()
       optimizer.step()
