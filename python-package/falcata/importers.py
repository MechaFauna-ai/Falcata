"""Import foreign gradient-boosting models into Falcata.

An imported model is an ordinary Falcata ``Booster``: it predicts through the
same code paths, serializes to the same formats (text or FALB), and can use
the GPU predictor. The point is to run one engine over models trained
anywhere.

Currently supported: XGBoost ``gbtree`` models with numeric splits
(:func:`from_xgboost`).

The conversions here are exact, not approximate, and the two places where the
engines genuinely disagree are handled explicitly rather than papered over:

* **Split comparison.** XGBoost sends a row left when ``x < threshold``;
  Falcata/LightGBM send it left when ``x <= threshold``. Converting the
  threshold to the next double DOWN makes ``x <= t'`` exactly equivalent to
  ``x < t``, because no double lies between them.
* **base_score.** XGBoost stores it in the objective's OUTPUT space (0.5 for
  ``binary:logistic``), while a tree ensemble sums raw margins. It is mapped
  back through the objective's inverse link and emitted as a constant first
  tree, so the sum is right without relying on any init-score plumbing.
  Multiclass carries one base_score PER CLASS; softmax is only shift-invariant
  under a uniform shift, so the per-class values are kept.
* **float32.** XGBoost is float32 throughout: its JSON stores the shortest
  decimal that round-trips a float32, and ``DMatrix`` truncates input data to
  float32. Thresholds and leaf values are therefore recovered THROUGH float32,
  not parsed as float64 -- otherwise a threshold lands a hair off the value
  XGBoost compares against and rows in the gap route differently.

  One consequence worth knowing: feed the converted model float64 data and it
  may route a row differently than XGBoost would with the SAME float64 data,
  because XGBoost first truncates it to float32. The converted model is the
  more faithful of the two there, but if you want identical answers, feed both
  float32 (which is what XGBoost sees anyway).
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .basic import Booster

__all__ = ["from_xgboost"]

# LightGBM decision_type bits (see include/Falcata/tree.h)
_CATEGORICAL_MASK = 1
_DEFAULT_LEFT_MASK = 2
_MISSING_TYPE_NAN = 2  # shifted into bits 2-3


def _objective_mapping(xgb_objective: str, num_class: int) -> Tuple[str, int, int, Any]:
    """(falcata objective string, num_class, num_tree_per_iteration, inverse link).

    The inverse link turns XGBoost's base_score -- which it stores in output
    space -- back into the raw margin the trees actually sum to.
    """
    if xgb_objective in ("reg:squarederror", "reg:linear", "reg:squaredlogerror"):
        if xgb_objective == "reg:squaredlogerror":
            raise ValueError("reg:squaredlogerror is not representable in Falcata")
        return "regression", 1, 1, (lambda p: p)
    if xgb_objective in ("binary:logistic", "reg:logistic"):
        def _logit(p: float) -> float:
            p = min(max(p, 1e-12), 1 - 1e-12)
            return math.log(p / (1.0 - p))
        return "binary sigmoid:1", 1, 1, _logit
    if xgb_objective in ("binary:logitraw",):
        # logitraw's predict() returns the raw margin, so the Falcata side must
        # NOT apply a sigmoid; regression is the identity-output objective
        return "regression", 1, 1, (lambda p: p)
    if xgb_objective in ("multi:softprob", "multi:softmax"):
        if num_class < 2:
            raise ValueError(f"{xgb_objective} needs num_class >= 2, got {num_class}")
        return f"multiclass num_class:{num_class}", num_class, num_class, (lambda p: p)
    if xgb_objective in ("count:poisson",):
        return "poisson", 1, 1, (lambda p: math.log(max(p, 1e-12)))
    raise ValueError(
        f"XGBoost objective '{xgb_objective}' is not supported by from_xgboost(); "
        "supported: reg:squarederror, binary:logistic, reg:logistic, "
        "binary:logitraw, multi:softprob, multi:softmax, count:poisson"
    )


def _xgb_json(model: Any) -> Dict[str, Any]:
    """Get the model JSON from a Booster, sklearn wrapper, path or dict."""
    if isinstance(model, dict):
        return model
    if isinstance(model, (str, Path)):
        path = Path(model)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # .ubj and legacy binaries need xgboost itself to decode
        try:
            import xgboost as xgb  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"reading '{path.name}' needs xgboost installed (only .json can be "
                "parsed without it)"
            ) from exc
        booster = xgb.Booster()
        booster.load_model(str(path))
        return json.loads(booster.save_raw("json").decode())
    # xgboost.Booster, or an sklearn wrapper holding one
    booster = getattr(model, "get_booster", None)
    if callable(booster):
        model = model.get_booster()
    if hasattr(model, "save_raw"):
        return json.loads(model.save_raw("json").decode())
    raise TypeError(f"cannot read an XGBoost model from {type(model)!r}")


class _TreeConverter:
    """One XGBoost tree -> the arrays a LightGBM text tree needs.

    XGBoost numbers every node in one array; LightGBM keeps internal nodes and
    leaves in separate arrays and encodes a leaf child as ``~leaf_index``. This
    walks the tree once, assigning both numberings.
    """

    def __init__(self, tree: Dict[str, Any]):
        self.left = tree["left_children"]
        self.right = tree["right_children"]
        # XGBoost is float32-native and its JSON stores the SHORTEST DECIMAL
        # that round-trips a float32. Parsing that decimal as float64 gives a
        # number a hair off the value XGBoost actually compares against
        # (0.0065392554 -> 0.0065392553999999992, while the real float32 is
        # 0.0065392553806304932), which silently reroutes rows that fall in
        # the gap. Round-trip through float32 to recover the exact value.
        self.cond = np.asarray(tree["split_conditions"], dtype=np.float32)
        self.feat = tree["split_indices"]
        self.default_left = tree["default_left"]
        self.gain = tree.get("loss_changes") or [0.0] * len(self.left)
        self.hess = tree.get("sum_hessian") or [0.0] * len(self.left)
        split_type = tree.get("split_type") or [0] * len(self.left)
        if any(int(s) != 0 for s in split_type):
            raise ValueError(
                "categorical splits are not supported by from_xgboost() yet "
                "(the model uses XGBoost's native categorical features)"
            )

        self.split_feature: List[int] = []
        self.threshold: List[float] = []
        self.decision_type: List[int] = []
        self.left_child: List[int] = []
        self.right_child: List[int] = []
        self.split_gain: List[float] = []
        self.internal_value: List[float] = []
        self.internal_count: List[int] = []
        self.leaf_value: List[float] = []
        self.leaf_count: List[int] = []
        self._n_internal = 0
        self._n_leaf = 0

    def _is_leaf(self, node: int) -> bool:
        return int(self.left[node]) == -1

    def convert(self) -> "_TreeConverter":
        if self._is_leaf(0):
            # a stump: LightGBM writes it as a single-leaf tree
            self.leaf_value.append(float(self.cond[0]))
            self.leaf_count.append(int(self.hess[0]))
            self._n_leaf = 1
            return self
        self._emit(0)
        return self

    def _emit(self, node: int) -> int:
        """Emit `node` (internal) and return its LightGBM internal index."""
        idx = self._n_internal
        self._n_internal += 1
        # placeholders, filled after the children are known
        self.split_feature.append(int(self.feat[node]))
        # XGBoost: go left when x < cond. LightGBM: go left when x <= t.
        # The next double DOWN makes the two identical -- nothing lies between.
        self.threshold.append(
            float(np.nextafter(np.float64(self.cond[node]), -np.inf)))
        dtype = _MISSING_TYPE_NAN << 2
        if int(self.default_left[node]):
            dtype |= _DEFAULT_LEFT_MASK
        self.decision_type.append(dtype)
        self.split_gain.append(float(self.gain[node]))
        self.internal_value.append(0.0)
        self.internal_count.append(int(self.hess[node]))
        self.left_child.append(0)
        self.right_child.append(0)

        for side, child in (("left", int(self.left[node])), ("right", int(self.right[node]))):
            if self._is_leaf(child):
                ref = ~self._n_leaf
                self._n_leaf += 1
                self.leaf_value.append(float(self.cond[child]))  # already exact f32
                self.leaf_count.append(int(self.hess[child]))
            else:
                ref = self._emit(child)
            if side == "left":
                self.left_child[idx] = ref
            else:
                self.right_child[idx] = ref
        return idx

    def to_text(self, shrinkage: float = 1.0) -> str:
        n_leaf = max(self._n_leaf, 1)
        if self._n_internal == 0:
            # LightGBM writes a single-leaf tree with empty node arrays
            return (
                "num_leaves=1\nnum_cat=0\nsplit_feature=\nsplit_gain=\nthreshold=\n"
                "decision_type=\nleft_child=\nright_child=\n"
                f"leaf_value={self.leaf_value[0]:.17g}\n"
                "leaf_weight=0\nleaf_count=0\ninternal_value=\ninternal_weight=\n"
                f"internal_count=\nis_linear=0\nshrinkage={shrinkage:.17g}\n"
            )

        def arr(vals, fmt="{}"):
            return " ".join(fmt.format(v) for v in vals)

        return (
            f"num_leaves={n_leaf}\n"
            "num_cat=0\n"
            f"split_feature={arr(self.split_feature)}\n"
            f"split_gain={arr(self.split_gain, '{:.17g}')}\n"
            f"threshold={arr(self.threshold, '{:.17g}')}\n"
            f"decision_type={arr(self.decision_type)}\n"
            f"left_child={arr(self.left_child)}\n"
            f"right_child={arr(self.right_child)}\n"
            f"leaf_value={arr(self.leaf_value, '{:.17g}')}\n"
            f"leaf_weight={arr([0] * n_leaf)}\n"
            f"leaf_count={arr(self.leaf_count)}\n"
            f"internal_value={arr(self.internal_value, '{:.17g}')}\n"
            f"internal_weight={arr([0] * self._n_internal)}\n"
            f"internal_count={arr(self.internal_count)}\n"
            "is_linear=0\n"
            f"shrinkage={shrinkage:.17g}\n"
        )


def _constant_tree(value: float) -> str:
    return (
        "num_leaves=1\nnum_cat=0\nsplit_feature=\nsplit_gain=\nthreshold=\n"
        "decision_type=\nleft_child=\nright_child=\n"
        f"leaf_value={value:.17g}\n"
        "leaf_weight=0\nleaf_count=0\ninternal_value=\ninternal_weight=\n"
        "internal_count=\nis_linear=0\nshrinkage=1\n"
    )


def from_xgboost(model: Any, feature_names: Optional[List[str]] = None) -> Booster:
    """Convert an XGBoost model into a Falcata :class:`Booster`.

    Parameters
    ----------
    model : xgboost.Booster, xgboost sklearn estimator, str, pathlib.Path or dict
        The model, a path to a saved model (``.json`` parses without xgboost
        installed; other formats need it), or an already-parsed model dict.
    feature_names : list of str or None, optional (default=None)
        Names for the converted model. Defaults to the names XGBoost carries,
        or ``Column_0 ...`` when it has none.

    Returns
    -------
    booster : Booster
        A Falcata Booster predicting the same function as the source model.

    Raises
    ------
    ValueError
        If the model uses a booster, objective or feature type that cannot be
        represented exactly (``gblinear``, ``dart``, native categorical splits,
        multi-output trees). The conversion refuses rather than approximating.

    Notes
    -----
    Predictions match XGBoost's ``output_margin=False`` output to float32
    rounding. Verified against XGBoost for regression, binary and multiclass
    objectives, including missing values (see tests/gates/import_xgboost.py).
    """
    raw = _xgb_json(model)
    learner = raw["learner"] if "learner" in raw else raw
    gb = learner["gradient_booster"]
    if gb.get("name") != "gbtree":
        raise ValueError(
            f"only XGBoost 'gbtree' models can be converted, got '{gb.get('name')}'"
        )
    mp = learner["learner_model_param"]
    num_feature = int(mp["num_feature"])
    num_class_raw = int(mp.get("num_class", 0) or 0)
    if int(mp.get("num_target", 1) or 1) != 1:
        raise ValueError("multi-target XGBoost trees are not supported")

    xgb_obj = learner["objective"]["name"]
    objective, num_class, trees_per_iter, inverse_link = _objective_mapping(
        xgb_obj, num_class_raw
    )

    base_score_raw = mp.get("base_score", "0")
    if isinstance(base_score_raw, str):
        base_score_raw = base_score_raw.strip().lstrip("[").rstrip("]")
        parts = [x for x in base_score_raw.split(",") if x.strip()]
    else:
        parts = list(base_score_raw) if isinstance(base_score_raw, (list, tuple)) else [base_score_raw]
    # multiclass carries one base_score PER CLASS; softmax is only shift
    # invariant under a uniform shift, so per-class values must be kept
    base_margins = [float(inverse_link(float(x))) for x in parts]
    if len(base_margins) == 1:
        base_margins = base_margins * trees_per_iter
    elif len(base_margins) != trees_per_iter:
        raise ValueError(
            f"base_score has {len(base_margins)} entries for {trees_per_iter} "
            "outputs; cannot map it"
        )

    trees = gb["model"]["trees"]
    tree_info = gb["model"].get("tree_info") or [0] * len(trees)
    if len(tree_info) != len(trees):
        tree_info = [0] * len(trees)

    # order trees iteration-major (class 0..k-1 within each iteration), which
    # is how LightGBM lays out a multiclass model
    order = sorted(range(len(trees)), key=lambda i: (i // max(trees_per_iter, 1), tree_info[i]))

    bodies: List[str] = []
    if any(v != 0.0 for v in base_margins):
        # a constant first iteration carries base_score into the sum
        for v in base_margins:
            bodies.append(_constant_tree(v))
    for i in order:
        bodies.append(_TreeConverter(trees[i]).convert().to_text())

    if feature_names is None:
        names = learner.get("feature_names") or []
        feature_names = list(names) if len(names) == num_feature else [
            f"Column_{i}" for i in range(num_feature)
        ]
    feature_names = [str(n).replace(" ", "_") for n in feature_names]

    chunks = []
    for i, body in enumerate(bodies):
        chunks.append(f"Tree={i}\n{body}\n")
    sizes = " ".join(str(len(c)) for c in chunks)

    header = (
        "tree\n"
        "version=v4\n"
        f"num_class={num_class}\n"
        f"num_tree_per_iteration={trees_per_iter}\n"
        "label_index=0\n"
        f"max_feature_idx={num_feature - 1}\n"
        f"objective={objective}\n"
        f"feature_names={' '.join(feature_names)}\n"
        f"feature_infos={' '.join(['none'] * num_feature)}\n"
        f"tree_sizes={sizes}\n\n"
    )
    text = header + "".join(chunks) + "end of trees\n"
    return Booster(model_str=text)
