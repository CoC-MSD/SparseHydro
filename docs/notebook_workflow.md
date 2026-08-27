# Notebook Workflow

This guide describes a safe way to run and modify the repository's Jupyter
notebooks without mixing experiments into the primary checkout or committing
notebook output accidentally.

## Checkout Layout

Keep the main checkout and notebook experiments as sibling directories:

```text
c:\dev\projects\SparseHydro\
c:\dev\projects\SparseHydro-notebooks\
```

The main checkout stays on `main`. Notebook work happens in the separate
`notebook-experiments` worktree and branch.

From `c:\dev\projects`, create the experiment worktree once:

```powershell
git -C .\SparseHydro worktree add -b notebook-experiments .\SparseHydro-notebooks main
```

If the worktree was accidentally created inside `SparseHydro`, move the
registered worktree rather than copying its files:

```powershell
git -C .\SparseHydro worktree move .\SparseHydro\SparseHydro-notebooks .\SparseHydro-notebooks
```

Check the arrangement with:

```powershell
git -C .\SparseHydro worktree list
```

## Create the Notebook Environment

Create one virtual environment inside the notebook worktree. The repository
already ignores `.venv/`, so it will not appear in Git changes.

```powershell
cd c:\dev\projects\SparseHydro-notebooks
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name sparsehydro-notebooks --display-name "Python (SparseHydro notebooks)"
```

Use the `Python (SparseHydro notebooks)` kernel in VS Code or Jupyter. The
editable install makes notebook imports use the code in this worktree, so
local package changes can be tested immediately.

`requirements.txt` is currently empty. Use `pyproject.toml` through the
editable install above. A Conda environment based on `environment.yml` is an
alternative if Conda is already part of your workflow.

## Running Notebooks

Open VS Code on `c:\dev\projects\SparseHydro-notebooks` when editing or
running notebooks. Start with notebooks that do not require external service
credentials:

- `sparsehydro/notebooks/dryweather.ipynb`
- `sparsehydro/notebooks/sheridan.ipynb`
- `sparsehydro/notebooks/unit_hydrograph_workflow.ipynb`

The Ayyeka and FlowFinity notebooks require their documented credentials and
appropriate data access. Keep credentials in the ignored local files or
environment variables described by those notebooks; never commit them.

## Keeping Output Out of Git

Notebook output is local experiment state. Before reviewing or sharing code,
clear outputs from the experiment worktree:

```powershell
cd c:\dev\projects\SparseHydro-notebooks
jupyter nbconvert --clear-output --inplace sparsehydro\notebooks\*.ipynb
```

Then review only the intended source changes:

```powershell
git diff --stat
git diff -- sparsehydro/notebooks
```

Commit notebook source changes on `notebook-experiments` only when they are
intentional. The main `SparseHydro` checkout is unaffected. To bring selected
work into the main branch, review and merge or cherry-pick the experiment
branch rather than copying an entire executed notebook with its outputs.

The generated HTML files under `sparsehydro/notebooks/artifacts/` are existing
visualization artifacts. Treat newly generated artifacts as local unless the
project explicitly needs them versioned.

## Returning to the Main Checkout

Deactivate the notebook environment before switching contexts:

```powershell
deactivate
cd c:\dev\projects\SparseHydro
git status --short --branch
```

Keep documentation and package changes intended for the shared project in the
main checkout. Keep exploratory notebook edits in the notebook worktree until
they are ready for review.
