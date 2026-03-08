"""
MlFinlab helps portfolio managers and traders who want to leverage the power of machine learning by providing
reproducible, interpretable, and easy to use tools.

Adding MlFinLab to your companies pipeline is like adding a department of PhD researchers to your team.
"""

"""Lightweight top-level imports.

Historically mlfinlab imported almost every submodule at package import time,
which makes `import mlfinlab.data_structures` fragile on modern Python/sklearn
setups due to optional/legacy dependencies in unrelated modules.

Keep the package import minimal and let users import submodules explicitly.
"""

from . import cross_validation
from . import data_structures
from . import datasets
from . import multi_product
from .filters import filters
from . import labeling
from .features import fracdiff
from . import sample_weights
from . import sampling
from . import bet_sizing
from . import util
from . import structural_breaks
from . import feature_importance
from . import clustering
from . import microstructural_features
from .backtest_statistics import backtests
from .backtest_statistics import statistics as backtest_statistics
from . import networks
from . import data_generation
from . import regression

# Ensemble has tight sklearn coupling across versions; keep optional.
try:
    from . import ensemble
except Exception:  # pragma: no cover
    ensemble = None
