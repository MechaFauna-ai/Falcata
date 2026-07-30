# coding: utf-8
"""Falcata, Light Gradient Boosting Machine.

Contributors: https://github.com/MechaFauna-ai/Falcata/graphs/contributors.
"""

from pathlib import Path

# .basic is intentionally loaded as early as possible, to dlopen() lib_falcata.{dll,dylib,so}
# and its dependencies as early as possible
from .basic import Booster, Dataset, Sequence, register_logger
from .callback import EarlyStopException, early_stopping, log_evaluation, record_evaluation, reset_parameter
from .engine import CVBooster, cv, train

try:
    from .sklearn import FalcataClassifier, FalcataModel, FalcataRanker, FalcataRegressor
except ImportError:
    pass
try:
    from .plotting import create_tree_digraph, plot_importance, plot_metric, plot_split_value_histogram, plot_tree
except ImportError:
    pass
try:
    from .dask import DaskFalcataClassifier, DaskFalcataRanker, DaskFalcataRegressor

    DaskLGBMRegressor = DaskFalcataRegressor
    DaskLGBMClassifier = DaskFalcataClassifier
    DaskLGBMRanker = DaskFalcataRanker
except ImportError:
    pass


_version_path = Path(__file__).resolve().parent / "VERSION.txt"
if _version_path.is_file():
    __version__ = _version_path.read_text(encoding="utf-8").strip()

__all__ = [
    "Dataset",
    "Booster",
    "CVBooster",
    "Sequence",
    "register_logger",
    "train",
    "cv",
    "FalcataModel",
    "LGBMModel",
    "FalcataRegressor",
    "LGBMRegressor",
    "FalcataClassifier",
    "LGBMClassifier",
    "FalcataRanker",
    "LGBMRanker",
    "DaskFalcataRegressor",
    "DaskLGBMRegressor",
    "DaskFalcataClassifier",
    "DaskLGBMClassifier",
    "DaskFalcataRanker",
    "DaskLGBMRanker",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
    "early_stopping",
    "EarlyStopException",
    "plot_importance",
    "plot_split_value_histogram",
    "plot_metric",
    "plot_tree",
    "create_tree_digraph",
]

# Backwards-compatible aliases for the pre-rename class names. Falcata's
# estimators are Falcata*; these keep code written against the LightGBM-era
# spelling working. They are the same objects, not subclasses.
LGBMModel = FalcataModel
LGBMRegressor = FalcataRegressor
LGBMClassifier = FalcataClassifier
LGBMRanker = FalcataRanker
