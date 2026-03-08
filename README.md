# fin-kit

Toolkit of quantitative finance utilities, built around `mlfinlab`, factor research, and agentic workflows.

## Prerequisites

- Python 3.9+
- `git`
- (Optional) `virtualenv`/`venv`

## Installation Guide

### Option A (recommended): one-command setup

Run the helper script to create a virtual environment, install dependencies, and place the package in editable mode:

```bash
./scripts/install_fin_kit.sh
```

The script accepts optional variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `PYTHON` | `python3` | Python interpreter used to create the venv |
| `VENV_DIR` | `.venv` inside the repo | Location of the virtual environment |

Example:

```bash
PYTHON=python3.11 VENV_DIR=$HOME/.venvs/fin-kit ./scripts/install_fin_kit.sh
```

After installation:

```bash
source .venv/bin/activate
python -m pip list  # verify packages
```

### Option B: manual installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Verify the install

Run a quick import check:

```bash
source .venv/bin/activate
python -c "import mlfinlab; print('fin-kit import ok')"
```

Then run a lightweight function test (optional):

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import pandas as pd

mod_path = Path('mlfinlab/filters/filters.py')
spec = importlib.util.spec_from_file_location('filters_mod', mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

idx = pd.date_range('2026-01-01', periods=10, freq='D')
prices = pd.Series([100,101,99,102,103,101,104,105,103,106], index=idx)
print(mod.cusum_filter(prices, threshold=1.5, time_stamps=True))
PY
```

## Running Tests

```bash
source .venv/bin/activate
pytest
```

## Using fin-kit inside other projects

Once installed, the package can be imported via:

```python
import mlfinlab
```

The repo is now ready to be consumed by tools such as `fruit-thrower` for MCP-based retrieval.
