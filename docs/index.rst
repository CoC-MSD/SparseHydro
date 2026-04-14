sparsehydro
===========

**sparsehydro** provides abstract interfaces and utilities for building
*parsimonious* (sparse-parameter) hydrological models in Python.

Features:

- Strict model lifecycle: ``CREATED → INITIALIZED → VALIDATED → PREPARED → PREDICTED → FINALIZED``
- Typed parameter registry with bounds for scalar and vector parameters
- Optional PyTorch integration for gradient-based calibration
- Sphinx documentation and GitHub Actions CI/CD out of the box

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
