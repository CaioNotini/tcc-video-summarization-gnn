---
inclusion: always
---

# Project files stay on drive D:

All project files for this repository stay on drive `D:`, and nothing is written to drive `C:`. This covers the virtual environment, the package caches, the model weights and the experiment outputs.

- The virtual environment is `.venv/` inside the repository (on `D:`). Do not create it elsewhere, and do not install packages into the system Python on `C:`.
- `PIP_CACHE_DIR` points to `.cache/pip/` (on `D:`), so pip never caches to `C:`.
- `TEMP` and `TMP` point to `.cache/tmp/` (on `D:`) while the venv is active, so build and install temporaries stay off `C:`.
- `TORCH_HOME` points to `.cache/torch/` (on `D:`), so torchvision and torch hub weights download there, not to `C:\Users\...\.cache\torch`.
- `MLFLOW_TRACKING_URI` is a file store at `.cache/mlruns/` (on `D:`).
- Frames, features, logs and experiment outputs already live under `data/` and `docs/evidence/`, both on `D:`.

These variables are set by the venv activation script (`.venv/Scripts/Activate.ps1` and `.venv/Scripts/activate.bat`), so they apply whenever the environment is activated. If the repository is moved or the venv is recreated, reapply the same variables to the new activation script.

When installing packages, rely on the activated `PIP_CACHE_DIR` (or pass `--no-cache-dir`) so nothing spills to `C:`. Before and after a large install, confirm the free space on `D:` and that the free space on `C:` did not decrease.
