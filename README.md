# fin-kit

Toolkit of quantitative finance utilities, built around `mlfinlab`, factor research, and agentic workflows.

## Prerequisites

- Python 3.9+
- `git`
- (Optional) `virtualenv`/`venv`

## Quick Installation

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

## Manual Installation (if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
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
