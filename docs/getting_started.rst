Getting Started
===============

.. contents:: Sections
   :local:
   :depth: 2

----

Installation
------------

Install the core package (requires Python 3.10+):

.. code-block:: bash

   pip install sparsehydro

Optional extras enable specific features:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Extra
     - Command
     - Enables
   * - ``torch``
     - ``pip install sparsehydro[torch]``
     - Gradient-based calibration via PyTorch
   * - ``rdii``
     - ``pip install sparsehydro[rdii]``
     - Physics-based RDII model (requires pymoo)
   * - ``platypus``
     - ``pip install sparsehydro[platypus]``
     - Full Platypus algorithm suite + PSO
   * - ``docs``
     - ``pip install sparsehydro[docs]``
     - Sphinx documentation build dependencies
   * - ``all``
     - ``pip install sparsehydro[all]``
     - Everything above

Interactive charts require ``plotly`` (not bundled, always install separately):

.. code-block:: bash

   pip install plotly

----

Quick Example
-------------

Implement a concrete model by subclassing :class:`~sparsehydro.interfaces.IModel`:

.. code-block:: python

   import pandas as pd
   from sparsehydro import IModel, ModelState, ScalarParameter

   class LinearReservoir(IModel):

       model_name = "linear-reservoir"

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

   model = LinearReservoir()
   model.initialize()
   model.validate()
   model.prepare(forcing=pd.Series([10.0, 8.0, 6.0], name="rainfall_mm"))
   output = model.predict()
   model.finalize()
   print(output)

The **model lifecycle** enforces that :meth:`~sparsehydro.interfaces.IModel.prepare`
is called before :meth:`~sparsehydro.interfaces.IModel.predict`, and
:meth:`~sparsehydro.interfaces.IModel.predict` before
:meth:`~sparsehydro.interfaces.IModel.finalize`.  Skipping steps raises a
:class:`~sparsehydro.interfaces.LifecycleError`.

----

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

----

Calibration Framework
---------------------

*sparsehydro* provides a solver-agnostic calibration engine.  The workflow is
always the same regardless of which solver you choose:

1. Build and prepare your model.
2. Create a :class:`~sparsehydro.calibration.problem.CalibrationProblem` that
   wraps the model, observed data, and one or more objectives.
3. Call ``solver.solve(problem)`` — returns a
   :class:`~sparsehydro.calibration.result.CalibrationResult`.
4. Inspect the Pareto front, per-generation history, and export to a DataFrame.

Defining a Calibration Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sparsehydro import (
       IModel, ModelState, ScalarParameter,
       CalibrationProblem, MSE, NashSutcliffe,
   )

   # --- Define a simple two-parameter model ---
   class LinearModel(IModel):
       model_name = "linear"

       def initialize(self) -> None:
           self.register_scalar_parameter(
               ScalarParameter("slope",     value=1.0, lower_bound=0.0, upper_bound=10.0)
           )
           self.register_scalar_parameter(
               ScalarParameter("intercept", value=0.0, lower_bound=-5.0, upper_bound=5.0)
           )
           self._state = ModelState.INITIALIZED

       def validate(self) -> bool:
           self._state = ModelState.VALIDATED
           return True

       def prepare(self, x: np.ndarray) -> None:
           self._x = x
           self._state = ModelState.PREPARED

       def predict(self) -> np.ndarray:
           slope     = self.get_scalar_parameter("slope").value
           intercept = self.get_scalar_parameter("intercept").value
           self._state = ModelState.PREDICTED
           return slope * self._x + intercept

       def finalize(self) -> None:
           self._state = ModelState.FINALIZED

   # --- Observed data ---
   x        = np.linspace(0, 10, 50)
   observed = 2.5 * x + 1.0 + np.random.default_rng(0).normal(0, 0.3, 50)

   # --- Prepare the model ---
   model = LinearModel()
   model.initialize()
   model.validate()
   model.prepare(x)

   # --- Wrap in a CalibrationProblem with two objectives ---
   problem = CalibrationProblem(
       model     = model,
       observed  = observed,
       objectives= [MSE(), NashSutcliffe()],
   )
   problem.prepare()

Objective Functions
~~~~~~~~~~~~~~~~~~~

All objectives implement :class:`~sparsehydro.calibration.objectives.IObjective`.
Minimisation and maximisation are handled automatically — the calibration engine
always minimises internally and converts back for display.

+-----------------------------------+--------------------------------------------+----------+
| Class                             | Measures                                   | Sense    |
+===================================+============================================+==========+
| :class:`~...objectives.MSE`       | Mean squared error                         | minimise |
+-----------------------------------+--------------------------------------------+----------+
| :class:`~...objectives.RMSE`      | Root mean squared error                    | minimise |
+-----------------------------------+--------------------------------------------+----------+
| :class:`~...objectives.MAE`       | Mean absolute error                        | minimise |
+-----------------------------------+--------------------------------------------+----------+
| :class:`~...objectives.PeakWeightedMSE` | MSE weighted by flow magnitude       | minimise |
+-----------------------------------+--------------------------------------------+----------+
| :class:`~...objectives.NashSutcliffe` | Nash-Sutcliffe efficiency (1 = perfect)| maximise |
+-----------------------------------+--------------------------------------------+----------+
| :class:`~...objectives.KGE`       | Kling-Gupta efficiency (1 = perfect)       | maximise |
+-----------------------------------+--------------------------------------------+----------+

Inspecting Results
~~~~~~~~~~~~~~~~~~

Every solver returns the same :class:`~sparsehydro.calibration.result.CalibrationResult`:

.. code-block:: python

   # After calling solver.solve(problem):
   result = solver.solve(problem)

   # Pareto front — shape (n_solutions, n_params)
   print(result.pareto_X)

   # Objective values in display form (maximised objectives un-negated)
   print(result.objective_display_values())

   # Full DataFrame with parameter values, objective values, and is_pareto flag
   df = result.to_pareto_dataframe()
   print(df.head())

   # Best solution by a specific objective
   best = result.best_by("MSE")
   print(best)       # {'slope': 2.49, 'intercept': 1.02, 'MSE': 0.087, ...}

----

Solvers
-------

All solvers share the same interface: construct with hyperparameters, then call
:meth:`~sparsehydro.calibration.solvers.base.ISolver.solve`.

NSGA-II (pymoo)
~~~~~~~~~~~~~~~

The default multi-objective solver.  Produces a true Pareto front across all
objectives.

.. code-block:: bash

   pip install pymoo

.. code-block:: python

   from sparsehydro import NSGAIISolver

   solver = NSGAIISolver(
       pop_size = 50,    # individuals per generation
       n_gen    = 100,   # number of generations
       seed     = 42,
   )
   result = solver.solve(problem)
   print(f"Pareto front: {len(result.pareto_X)} solutions")

SciPy Solver
~~~~~~~~~~~~

Single-objective wrapper around :mod:`scipy.optimize`.  Use when you only
have one objective and want a fast, gradient-free or gradient-based search.

.. code-block:: bash

   pip install scipy

.. code-block:: python

   from sparsehydro import ScipySolver

   # Differential Evolution — global, gradient-free
   solver = ScipySolver(
       method          = "differential_evolution",
       objective_index = 0,   # which objective to minimise (index into objectives list)
       maxiter         = 300,
       seed            = 42,
   )
   result = solver.solve(problem)
   print(result.best_by("MSE"))

   # Nelder-Mead — local, gradient-free
   solver = ScipySolver(method="Nelder-Mead", objective_index=0)
   result = solver.solve(problem)

Platypus Solver
~~~~~~~~~~~~~~~

Pass **any** Platypus algorithm class and its constructor arguments.  The full
Platypus suite — NSGA-II, NSGA-III, SPEA2, MOEA/D, GDE3, IBEA, ε-MOEA — is
accessible through a single solver class.

.. code-block:: bash

   pip install platypus-opt

.. code-block:: python

   import platypus
   from sparsehydro import PlatypusSolver

   # NSGA-II via Platypus
   solver = PlatypusSolver(
       platypus.NSGAII,
       population_size = 50,
       n_evaluations   = 5_000,
       seed            = 42,
   )
   result = solver.solve(problem)

   # SPEA2
   solver = PlatypusSolver(platypus.SPEA2, population_size=50, n_evaluations=5_000)

   # GDE3 — good for real-valued many-objective problems
   solver = PlatypusSolver(platypus.GDE3, population_size=80, n_evaluations=8_000)

   # ε-MOEA — maintains an ε-dominance archive
   solver = PlatypusSolver(
       platypus.EpsMOEA,
       epsilons        = [0.01, 0.01],   # one per objective
       population_size = 100,
       n_evaluations   = 10_000,
   )

Particle Swarm (PSO)
~~~~~~~~~~~~~~~~~~~~

Wraps Platypus's Speed-constrained Multi-objective PSO (**SMPSO**) by default.
Supply ``epsilons`` to switch to **OMOPSO**, which maintains an ε-dominance
archive instead of a standard Pareto archive.

.. code-block:: python

   from sparsehydro import ParticleSwarmSolver

   # SMPSO — default
   solver = ParticleSwarmSolver(
       swarm_size    = 50,
       n_evaluations = 5_000,
       seed          = 42,
   )
   result = solver.solve(problem)

   # OMOPSO — ε-dominance archive; supply one epsilon per objective
   solver = ParticleSwarmSolver(
       swarm_size    = 50,
       n_evaluations = 5_000,
       epsilons      = [0.01, 0.01],
   )
   result = solver.solve(problem)

Choosing a Solver
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - Solver
     - Objectives
     - Speed
     - When to use
   * - :class:`NSGAIISolver`
     - 2–3
     - Medium
     - Standard multi-objective baseline; well-studied
   * - :class:`ScipySolver`
     - 1
     - Fast
     - Quick single-objective runs; local or global scipy methods
   * - :class:`PlatypusSolver` (SPEA2 / GDE3)
     - 2–4
     - Medium
     - Exploring different multi-objective algorithms with the same API
   * - :class:`PlatypusSolver` (ε-MOEA)
     - 2–5
     - Medium
     - Sparse ε-dominance archive; good for many objectives
   * - :class:`ParticleSwarmSolver` (SMPSO)
     - 2–3
     - Fast
     - Continuous search spaces; fast convergence
   * - :class:`ParticleSwarmSolver` (OMOPSO)
     - 2–4
     - Fast
     - PSO with ε-archive; avoids crowded Pareto fronts

----

Interactive Visualization
--------------------------

All visualization functions require ``plotly``:

.. code-block:: bash

   pip install plotly

Every function returns a :class:`plotly.graph_objects.Figure` that can be
displayed in a Jupyter notebook with ``.show()``, embedded in a web app, or
written to an HTML file with ``.write_html("output.html")``.

Time-Series Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

**plot_timeseries** — rainfall (inverted bar) + observed/predicted flow:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sparsehydro import plot_timeseries

   dt   = pd.date_range("2020-01-01", periods=120, freq="h")
   rain = np.clip(np.random.default_rng(0).exponential(1.5, 120), 0, None)
   obs  = np.sin(np.linspace(0, 4*np.pi, 120)) * 5 + 10
   pred = obs + np.random.default_rng(1).normal(0, 0.5, 120)

   fig = plot_timeseries(dt, rain, obs, pred, title="My Model Run")
   fig.show()
   fig.write_html("timeseries.html")

**plot_residuals_scatter** — three-panel residual diagnostics:

.. code-block:: python

   from sparsehydro import plot_residuals_scatter

   fig = plot_residuals_scatter(
       dt, obs, pred,
       title      = "Residual Diagnostics",
       flow_label = "Flow [m³/s]",
   )
   fig.show()

The three panels are:

1. Observed vs predicted scatter with a 1:1 diagonal line.
2. Residual (obs − pred) bar chart coloured red/blue for over/under prediction.
3. Residual autocorrelation at lags 0–30 with 95 % confidence bands.

**plot_cumulative_volume** — cumulative volume balance:

.. code-block:: python

   from sparsehydro import plot_cumulative_volume

   fig = plot_cumulative_volume(dt, obs, pred, title="Volume Balance")
   fig.show()

Calibration Result Plots
~~~~~~~~~~~~~~~~~~~~~~~~

All functions below accept a
:class:`~sparsehydro.calibration.result.CalibrationResult` returned by any
solver.

**plot_pareto_evolution** — animated Pareto front with Play/Pause and a
generation slider:

.. code-block:: python

   from sparsehydro import plot_pareto_evolution

   fig = plot_pareto_evolution(
       result,
       x_obj = "MSE",           # objective name or 0-based index
       y_obj = "NashSutcliffe",
       title = "Pareto Evolution",
   )
   fig.show()

**plot_parallel_coordinates** — draggable parallel-axis explorer.  Each line
is one solution; drag axis endpoints to filter:

.. code-block:: python

   from sparsehydro import plot_parallel_coordinates

   fig = plot_parallel_coordinates(
       result,
       color_by             = "NashSutcliffe",
       use_final_pareto_only= True,
   )
   fig.show()

**plot_objective_convergence** — best-value line + 10th–90th percentile band
per generation for each objective:

.. code-block:: python

   from sparsehydro import plot_objective_convergence

   fig = plot_objective_convergence(result, title="Convergence")
   fig.show()

**plot_parameter_distributions** — violin plots showing the spread of each
calibrated parameter across the Pareto front:

.. code-block:: python

   from sparsehydro import plot_parameter_distributions

   fig = plot_parameter_distributions(result, use_final_pareto_only=True)
   fig.show()

**plot_sensitivity_heatmap** — Pearson correlation heatmap between parameters
(columns) and objectives (rows).  High absolute values indicate influential
parameters:

.. code-block:: python

   from sparsehydro import plot_sensitivity_heatmap

   fig = plot_sensitivity_heatmap(result, title="Parameter Sensitivity")
   fig.show()

**plot_pareto_scatter_matrix** — scatter-plot matrix (SPLOM) of all objective
pairs.  Useful when there are three or more objectives:

.. code-block:: python

   from sparsehydro import plot_pareto_scatter_matrix

   fig = plot_pareto_scatter_matrix(result, title="Objective Trade-offs")
   fig.show()

RDII-Specific Plots
~~~~~~~~~~~~~~~~~~~

These functions require ``sparsehydro[rdii]``.

**plot_rtk_shape** — unit hydrograph shape for each
:class:`~sparsehydro.rdii.rtk_triangle.RTKTriangle`, useful for
sanity-checking T (time-to-peak) and K (recession ratio) before calibration:

.. code-block:: python

   from sparsehydro.rdii import RTKTriangle
   from sparsehydro import plot_rtk_shape

   tri1 = RTKTriangle(R=0.05, T=1.0, K=2.0)
   tri1.initialize(); tri1.validate()
   tri2 = RTKTriangle(R=0.02, T=3.0, K=3.5)
   tri2.initialize(); tri2.validate()

   fig = plot_rtk_shape([tri1, tri2], dt_hours=0.25)
   fig.show()

**plot_rdii_components** — stacked area chart of per-triangle RDII
contributions alongside rainfall.  The input DataFrame must contain
``rdii_component_N`` columns (produced by
:class:`~sparsehydro.rdii.model.RDIIModel`):

.. code-block:: python

   from sparsehydro import plot_rdii_components

   # result_df is the DataFrame returned by RDIIModel.predict()
   fig = plot_rdii_components(result_df, title="RDII Components")
   fig.show()

Multi-Panel Dashboard
~~~~~~~~~~~~~~~~~~~~~

:func:`~sparsehydro.visualization.dashboard.plot_calibration_dashboard`
assembles up to five panels into a single, fully self-contained HTML file —
no server required:

.. code-block:: python

   from sparsehydro import plot_calibration_dashboard

   fig = plot_calibration_dashboard(
       result,
       timeseries_df = ts_df,            # optional — adds a time-series panel
       observed_col  = "observed",
       rainfall_col  = "rainfall_mm",
       flow_label    = "Flow [m³/s]",
       title         = "My Calibration Dashboard",
       output_path   = "dashboard.html", # None = skip writing
       use_cdn       = True,             # False = inline Plotly JS (~4 MB)
   )
   # Returns the convergence figure for interactive notebook use
   fig.show()

Opening ``dashboard.html`` in any browser shows all panels:

1. Time series (if *timeseries_df* provided)
2. Objective convergence
3. Animated Pareto front evolution
4. Parameter distributions
5. Parallel coordinates

VisualizationModel
~~~~~~~~~~~~~~~~~~

:class:`~sparsehydro.visualization.timeseries.VisualizationModel` wraps the
three core plot functions behind the standard
:class:`~sparsehydro.interfaces.IModel` lifecycle so a visualization step can
be composed into any pipeline:

.. code-block:: python

   from sparsehydro import VisualizationModel

   viz = VisualizationModel(title="RDII Calibration Run")
   viz.initialize()
   viz.validate()
   viz.prepare(
       result_df,
       datetime_col       = "datetime",
       predicted_col      = "rdii_mm",
       observed_col       = "flow_cfs",
       rainfall_col       = "rainfall_mm",
       calibration_result = result,         # optional
   )
   viz.predict()

   viz.timeseries_figure.show()   # always generated
   viz.pareto_figure.show()       # requires calibration_result
   viz.parallel_figure.show()     # requires calibration_result

   viz.finalize()

----

RDII Modelling
--------------

The :mod:`sparsehydro.rdii` subpackage provides a physics-based RDII model
that combines temperature-driven **initial abstraction** recovery with
triangular **RTK unit hydrographs**.

.. code-block:: bash

   pip install sparsehydro[rdii]

Quick RDII calibration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from sparsehydro.rdii import RDIIOptimizer

   # ts must contain columns: datetime, rainfall_mm, temperature_c, observed_flow
   ts = pd.read_csv("my_gauge_data.csv", parse_dates=["datetime"])

   optimizer = RDIIOptimizer(n_triangles=3)
   result    = optimizer.calibrate(
       ts,
       datetime_col  = "datetime",
       rainfall_col  = "rainfall_mm",
       temp_col      = "temperature_c",
       observed_col  = "observed_flow",
       pop_size      = 50,
       n_gen         = 100,
   )

   # Best parameters by Nash-Sutcliffe efficiency
   best = result.best_by("NashSutcliffe")
   print(best)

   # Visualise
   from sparsehydro import plot_timeseries, plot_pareto_evolution
   plot_timeseries(...).show()
   plot_pareto_evolution(result).show()

See the :ref:`API Reference <api-reference>` for complete documentation of
:class:`~sparsehydro.rdii.model.RDIIModel`,
:class:`~sparsehydro.rdii.initial_abstraction.InitialAbstractionModel`,
:class:`~sparsehydro.rdii.rtk_triangle.RTKTriangle`, and
:class:`~sparsehydro.rdii.optimization.RDIIOptimizer`.
