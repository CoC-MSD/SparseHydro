API Reference
=============

.. contents:: Sections
   :local:
   :depth: 2

----

Core Package
------------

Top-level package
~~~~~~~~~~~~~~~~~

The top-level :mod:`sparsehydro` namespace re-exports the most commonly used
names so that a single import covers the essential API:

.. code-block:: python

   from sparsehydro import (
       IModel, ModelState, ScalarParameter,
       CalibrationProblem, CalibrationResult,
       MSE, RMSE, NashSutcliffe,
       NSGAIISolver, ScipySolver, PlatypusSolver, ParticleSwarmSolver,
       plot_timeseries, plot_calibration_dashboard,
   )

.. automodule:: sparsehydro
   :members:
   :show-inheritance:
   :noindex:

Enumerations
~~~~~~~~~~~~

.. automodule:: sparsehydro.enums
   :members:
   :show-inheritance:
   :undoc-members:

Parameters
~~~~~~~~~~

Parameters are the connective tissue between the model physics and the
calibration engine.  Every numeric quantity that may be optimised must be
registered with the model via a :class:`~sparsehydro.parameters.ScalarParameter`
(single value) or :class:`~sparsehydro.parameters.VectorParameter` (array).

.. automodule:: sparsehydro.parameters
   :members:
   :show-inheritance:

Model Interface
~~~~~~~~~~~~~~~

All physical models in *sparsehydro* implement :class:`~sparsehydro.interfaces.IModel`.
The lifecycle enforces a consistent prepare → predict → finalize pattern and
keeps the calibration engine decoupled from model internals.

.. automodule:: sparsehydro.interfaces
   :members:
   :show-inheritance:

Model Registry
~~~~~~~~~~~~~~

.. automodule:: sparsehydro.registry
   :members:
   :show-inheritance:

PyTorch Interface
~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.torch_model
   :members:
   :show-inheritance:

----

Visualization
-------------

The :mod:`sparsehydro.visualization` package provides interactive, browser-ready
Plotly charts for time-series diagnostics, calibration result exploration, and
RDII component analysis.  All functions require the optional ``plotly`` dependency:

.. code-block:: bash

   pip install plotly

All 13 public names are importable directly from :mod:`sparsehydro.visualization`
or from the top-level :mod:`sparsehydro` namespace.

.. automodule:: sparsehydro.visualization
   :members:
   :show-inheritance:
   :noindex:

Time-Series Plots
~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.timeseries
   :members:
   :show-inheritance:

Calibration Result Plots
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.calibration
   :members:
   :show-inheritance:

RDII-Specific Plots
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.rdii
   :members:
   :show-inheritance:

Calibration Dashboard
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.visualization.dashboard
   :members:
   :show-inheritance:

----

Unit Hydrograph
---------------

.. automodule:: sparsehydro.unithydrograph
   :members:
   :show-inheritance:
   :noindex:

Adapter
~~~~~~~

.. automodule:: sparsehydro.unithydrograph.adapter
   :members:
   :show-inheritance:

----

RDII
----

The :mod:`sparsehydro.rdii` subpackage implements the full physics-based
Rainfall-Derived Inflow and Infiltration (RDII) modelling chain:

* Initial abstraction with temperature-dependent recovery
* Triangular RTK unit hydrographs
* Multi-objective calibration objectives
* One-stop :class:`~sparsehydro.rdii.optimization.RDIIOptimizer`

Requires the optional ``rdii`` extra:

.. code-block:: bash

   pip install sparsehydro[rdii]

.. automodule:: sparsehydro.rdii
   :members:
   :show-inheritance:
   :noindex:

RDII Model
~~~~~~~~~~

.. automodule:: sparsehydro.rdii.model
   :members:
   :show-inheritance:

Initial Abstraction
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.rdii.initial_abstraction
   :members:
   :show-inheritance:

RTK Triangle
~~~~~~~~~~~~

.. automodule:: sparsehydro.rdii.rtk_triangle
   :members:
   :show-inheritance:

RDII Objectives
~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.rdii.objectives
   :members:
   :show-inheritance:

RDII Optimization
~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.rdii.optimization
   :members:
   :show-inheritance:

RDII Visualization (shim)
~~~~~~~~~~~~~~~~~~~~~~~~~

Re-exports :func:`~sparsehydro.visualization.plot_timeseries`,
:func:`~sparsehydro.visualization.plot_pareto_evolution`, and
:func:`~sparsehydro.visualization.plot_parallel_coordinates` for
backward compatibility.

.. automodule:: sparsehydro.rdii.visualization
   :members:
   :show-inheritance:

----

Calibration
-----------

The :mod:`sparsehydro.calibration` subpackage provides solver-agnostic
abstractions for parameter estimation:

* :class:`~sparsehydro.calibration.problem.CalibrationProblem` — wraps any
  :class:`~sparsehydro.interfaces.IModel` with observed data and objectives.
* :class:`~sparsehydro.calibration.result.CalibrationResult` — stores the
  Pareto front, per-generation history, and metadata.
* A library of :class:`~sparsehydro.calibration.objectives.IObjective`
  implementations covering MSE, NSE, KGE, and more.
* Multiple solver backends (see :ref:`Solvers` below).

.. automodule:: sparsehydro.calibration
   :members:
   :show-inheritance:
   :noindex:

Calibration Problem
~~~~~~~~~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.problem
   :members:
   :show-inheritance:

Calibration Result
~~~~~~~~~~~~~~~~~~

:class:`~sparsehydro.calibration.result.CalibrationResult` stores objective
values in **minimisation form** internally: maximised objectives (e.g. NSE,
KGE) are negated on the way in and restored via
:meth:`~sparsehydro.calibration.result.CalibrationResult.objective_display_values`
and :meth:`~sparsehydro.calibration.result.CalibrationResult.to_pareto_dataframe`.

.. automodule:: sparsehydro.calibration.result
   :members:
   :show-inheritance:

Objectives
~~~~~~~~~~

+-------------------------------+---------------------------------------------------+
| Class                         | Description                                       |
+===============================+===================================================+
| :class:`MSE`                  | Mean squared error (minimise)                     |
+-------------------------------+---------------------------------------------------+
| :class:`RMSE`                 | Root mean squared error (minimise)                |
+-------------------------------+---------------------------------------------------+
| :class:`MAE`                  | Mean absolute error (minimise)                    |
+-------------------------------+---------------------------------------------------+
| :class:`PeakWeightedMSE`      | Flow-weighted MSE — penalises peak-flow errors    |
+-------------------------------+---------------------------------------------------+
| :class:`NashSutcliffe`        | Nash-Sutcliffe efficiency (maximise)              |
+-------------------------------+---------------------------------------------------+
| :class:`KGE`                  | Kling-Gupta efficiency (maximise)                 |
+-------------------------------+---------------------------------------------------+

.. automodule:: sparsehydro.calibration.objectives
   :members:
   :show-inheritance:

----

.. _Solvers:

Solvers
-------

All solvers implement :class:`~sparsehydro.calibration.solvers.base.ISolver`
and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

.. list-table::
   :header-rows: 1
   :widths: 35 40 25

   * - Class
     - Algorithm
     - Extra dep
   * - :class:`~sparsehydro.calibration.solvers.nsga2.NSGAIISolver`
     - NSGA-II (pymoo)
     - ``pymoo``
   * - :class:`~sparsehydro.calibration.solvers.scipy_solver.ScipySolver`
     - SciPy minimisers
     - ``scipy``
   * - :class:`~sparsehydro.calibration.solvers.platypus_solver.PlatypusSolver`
     - Any Platypus algorithm
     - ``platypus-opt``
   * - :class:`~sparsehydro.calibration.solvers.pso_solver.ParticleSwarmSolver`
     - SMPSO / OMOPSO (PSO)
     - ``platypus-opt``

.. automodule:: sparsehydro.calibration.solvers
   :members:
   :show-inheritance:
   :noindex:

Solver Base
~~~~~~~~~~~

.. automodule:: sparsehydro.calibration.solvers.base
   :members:
   :show-inheritance:

NSGA-II Solver
~~~~~~~~~~~~~~

Requires ``pymoo``:

.. code-block:: bash

   pip install pymoo

.. automodule:: sparsehydro.calibration.solvers.nsga2
   :members:
   :show-inheritance:

SciPy Solver
~~~~~~~~~~~~

Requires ``scipy``:

.. code-block:: bash

   pip install scipy

.. automodule:: sparsehydro.calibration.solvers.scipy_solver
   :members:
   :show-inheritance:

Platypus Solver
~~~~~~~~~~~~~~~

Wraps **any** :class:`platypus.Algorithm` subclass — NSGA-II, NSGA-III, SPEA2,
MOEA/D, GDE3, IBEA, ε-MOEA, and more — behind the uniform
:class:`~sparsehydro.calibration.solvers.base.ISolver` interface.

Requires ``platypus-opt``:

.. code-block:: bash

   pip install platypus-opt

.. automodule:: sparsehydro.calibration.solvers.platypus_solver
   :members:
   :show-inheritance:

Particle Swarm Solver
~~~~~~~~~~~~~~~~~~~~~

Wraps :class:`platypus.SMPSO` (default) or :class:`platypus.OMOPSO`
(when ``epsilons`` are provided).

Requires ``platypus-opt``:

.. code-block:: bash

   pip install platypus-opt

.. automodule:: sparsehydro.calibration.solvers.pso_solver
   :members:
   :show-inheritance:
