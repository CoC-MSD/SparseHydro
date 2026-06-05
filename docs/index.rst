.. include:: ../README.md
   :parser: myst_parser.sphinx_

----

Package Architecture
--------------------

The diagram below shows how the core abstractions relate to each other.
Models expose named :class:`~sparsehydro.parameters.ScalarParameter` objects
(each with an optional ``calibrate`` flag) which are discovered automatically
by :class:`~sparsehydro.calibration.problem.CalibrationProblem`.  A single
problem instance is passed unchanged to any solver backend.

.. mermaid::
   :caption: sparsehydro component architecture

   graph TB
       subgraph core["Core — sparsehydro"]
           IM["IModel\n(abstract lifecycle)"]
           SP["ScalarParameter\nvalue · bounds · calibrate"]
           MS["ModelState\nCREATED → FINALIZED"]
       end

       subgraph models["Concrete Models"]
           RDII["RDIIModel\nIA + N × RTK triangles"]
           CHM["CombinedHydroModel\nany IA + any UH mix"]
           UHA["UnitHydrographAdapter"]
           CUSTOM["Custom Model\n(subclass IModel)"]
       end

       subgraph calibration["Calibration — sparsehydro.calibration"]
           CP["CalibrationProblem\ndata · column_map · objectives"]
           OBJ["Objectives\nNSE · KGE · PeakWMSE · MSE"]
           IC["InequalityConstraints\ng_j ≤ 0"]

           subgraph solvers["Solvers  (ISolver)"]
               NS["NSGAIISolver\npymoo"]
               PS["ParticleSwarmSolver\nSMPSO / OMOPSO"]
               SC["ScipySolver\ndiff-evo / L-BFGS-B"]
               PL["PlatypusSolver\nany Platypus algo"]
           end

           CR["CalibrationResult\npareto_X · pareto_F · history"]
       end

       subgraph viz["Visualization — sparsehydro.visualization"]
           V1["plot_calibration_timeseries\n2-row dashboard + scatter"]
           V2["plot_pareto_evolution"]
           V3["plot_parallel_coordinates"]
           V4["plot_calibration_dashboard"]
       end

       IM --> RDII & CHM & UHA & CUSTOM
       IM -- "registers" --> SP
       RDII & UHA & CUSTOM --> CP
       OBJ --> CP
       IC --> CP
       CP --> NS & PS & SC & PL
       NS & PS & SC & PL --> CR
       CR --> V1 & V2 & V3 & V4

       style core fill:#e8f4fd,stroke:#2c5f99
       style models fill:#eafaf1,stroke:#1e8449
       style calibration fill:#fef9e7,stroke:#d4ac0d
       style viz fill:#fdedec,stroke:#c0392b

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   api
