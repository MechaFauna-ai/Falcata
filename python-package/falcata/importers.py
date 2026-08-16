"""Import foreign gradient-boosting models into Falcata.

An imported model is an ordinary Falcata ``Booster``: it predicts through the
same code paths, serializes to the same formats (text or FALB), and can use
the GPU predictor. The point is to run one engine over models trained
anywhere.

Currently supported: XGBoost ``gbtree`` models with numeric splits
(:func:`from_xgboost`) and CatBoost models over numeric features
(:func:`from_catboost`).

The conversions here are exact, not approximate, and the two places where the
engines genuinely disagree are handled explicitly rather than papered over:

* **Split comparison.** XGBoost sends a row left when ``x < threshold``;
  Falcata/LightGBM send it left when ``x <= threshold``. Converting the
  threshold to the next FLOAT32 down makes ``x <= t'`` exactly equivalent to
  ``x < t`` -- no float32 lies between them, and XGBoost compares in float32
  anyway. The step stays on the float32 grid so that single-precision
  evaluation of the converted model agrees with double-precision evaluation.
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

__all__ = ["from_catboost", "from_xgboost"]

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
                f"reading '{path.name}' needs xgboost installed (only .json can be parsed without it)"
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
        # The next FLOAT32 down makes the two identical for float32 data --
        # nothing representable lies between -- and, unlike the next double
        # down, it is itself a float32. That matters because the model is also
        # evaluated in single precision (FIL on the GPU, via treelite): a
        # double-space step rounds back to cond there, so every row sitting
        # exactly on a split -- and splits are chosen AT data values, so there
        # are many -- took the other branch. It cost 3.6% of rows on an
        # imported model, against 0.01% for a natively trained one.
        self.threshold.append(float(np.nextafter(self.cond[node], np.float32("-inf"))))
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

        def arr(vals: Any, fmt: str = "{}") -> str:
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
    learner = raw.get("learner", raw)
    gb = learner["gradient_booster"]
    if gb.get("name") != "gbtree":
        raise ValueError(f"only XGBoost 'gbtree' models can be converted, got '{gb.get('name')}'")
    mp = learner["learner_model_param"]
    num_feature = int(mp["num_feature"])
    num_class_raw = int(mp.get("num_class", 0) or 0)
    if int(mp.get("num_target", 1) or 1) != 1:
        raise ValueError("multi-target XGBoost trees are not supported")

    xgb_obj = learner["objective"]["name"]
    objective, num_class, trees_per_iter, inverse_link = _objective_mapping(xgb_obj, num_class_raw)

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
        raise ValueError(f"base_score has {len(base_margins)} entries for {trees_per_iter} outputs; cannot map it")

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
        feature_names = list(names) if len(names) == num_feature else [f"Column_{i}" for i in range(num_feature)]
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


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


def _catboost_json(model: Any) -> Dict[str, Any]:
    """Get the model JSON from a CatBoost model, a path, or a parsed dict."""
    if isinstance(model, dict):
        return model
    if isinstance(model, (str, Path)):
        path = Path(model)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        try:
            import catboost  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"reading '{path.name}' needs catboost installed (only .json can be parsed without it)"
            ) from exc
        loaded = catboost.CatBoost()
        loaded.load_model(str(path))
        model = loaded
    if hasattr(model, "save_model"):
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "model.json")
            model.save_model(out, format="json")
            with open(out, "r", encoding="utf-8") as f:
                return json.load(f)
    raise TypeError(f"cannot read a CatBoost model from {type(model)!r}")


def _catboost_objective(raw: Dict[str, Any]) -> Tuple[str, Any]:
    """(falcata objective string, leaf transform) for a CatBoost loss."""
    info = raw.get("model_info", {}) or {}
    params = info.get("params")
    loss = ""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    if isinstance(params, dict):
        loss = str(
            (params.get("loss_function") or {}).get("type")
            if isinstance(params.get("loss_function"), dict)
            else params.get("loss_function", "")
        )
    if loss in ("RMSE", "", "Quantile", "MAE", "LossFunctionChange"):
        return "regression", None
    if loss == "Logloss":
        return "binary sigmoid:1", None
    if loss == "Poisson":
        return "poisson", None
    raise ValueError(
        f"CatBoost loss_function '{loss}' is not supported by from_catboost(); "
        "supported: RMSE, MAE, Quantile, Logloss, Poisson"
    )


def _oblivious_to_text(
    splits: List[Dict[str, Any]], leaf_values: List[float], leaf_weights: Optional[List[float]], scale: float
) -> str:
    """One CatBoost oblivious (symmetric) tree -> a LightGBM text tree.

    CatBoost picks a leaf by bits: ``leaf = sum((x[f_i] > border_i) << i)`` over
    the splits in listed order. A perfect binary tree reproduces that exactly if
    the ROOT tests the LAST split and each level down tests the next-lower one,
    because then the left-to-right leaf position is the same bit number.

    The split sense also lines up without adjusting the threshold: CatBoost
    sends ``x > border`` to bit 1, and LightGBM sends ``x <= threshold`` left,
    so bit 0 is the left child at threshold == border.
    """
    depth = len(splits)
    n_leaf = 1 << depth
    n_internal = n_leaf - 1
    split_feature, threshold, decision_type = [], [], []
    left_child, right_child = [], []
    # level-order (BFS) numbering: node k at level L, children 2k+1 / 2k+2
    for node in range(n_internal):
        level = int(math.floor(math.log2(node + 1)))
        sp = splits[depth - 1 - level]
        split_feature.append(int(sp["float_feature_index"]))
        threshold.append(float(np.float64(sp["border"])))
        # NaN routing comes from the feature's nan_value_treatment, resolved by
        # the caller into "nan_left"
        decision_type.append((_MISSING_TYPE_NAN << 2) | (_DEFAULT_LEFT_MASK if sp.get("nan_left", True) else 0))
        lc, rc = 2 * node + 1, 2 * node + 2
        left_child.append(lc if lc < n_internal else ~(lc - n_internal))
        right_child.append(rc if rc < n_internal else ~(rc - n_internal))
    weights = leaf_weights or [0] * n_leaf

    def arr(vals: Any, fmt: str = "{}") -> str:
        return " ".join(fmt.format(v) for v in vals)

    return (
        f"num_leaves={n_leaf}\n"
        "num_cat=0\n"
        f"split_feature={arr(split_feature)}\n"
        f"split_gain={arr([0] * n_internal)}\n"
        f"threshold={arr(threshold, '{:.17g}')}\n"
        f"decision_type={arr(decision_type)}\n"
        f"left_child={arr(left_child)}\n"
        f"right_child={arr(right_child)}\n"
        f"leaf_value={arr([v * scale for v in leaf_values], '{:.17g}')}\n"
        f"leaf_weight={arr([0] * n_leaf)}\n"
        f"leaf_count={arr([int(w) for w in weights])}\n"
        f"internal_value={arr([0] * n_internal)}\n"
        f"internal_weight={arr([0] * n_internal)}\n"
        f"internal_count={arr([0] * n_internal)}\n"
        "is_linear=0\n"
        "shrinkage=1\n"
    )


def from_catboost(model: Any, feature_names: Optional[List[str]] = None) -> Booster:
    """Convert a CatBoost model over numeric features into a Falcata Booster.

    Parameters
    ----------
    model : catboost model, str, pathlib.Path or dict
        A CatBoost estimator, a path to a saved model (``.json`` parses without
        catboost installed; other formats need it), or a parsed model dict.
    feature_names : list of str or None, optional (default=None)
        Names for the converted model; defaults to CatBoost's own, else
        ``Column_0 ...``.

    Returns
    -------
    booster : Booster
        A Falcata Booster predicting the same function as the source model.

    Raises
    ------
    ValueError
        For models this cannot represent exactly: categorical/CTR features,
        multiclass, or a non-oblivious structure. It refuses rather than
        approximating.

    Notes
    -----
    CatBoost's symmetric trees are unrolled into ordinary binary trees, so the
    converted model is larger in node count but predicts identically (verified
    to ~1e-6 relative; see tests/gates/import_catboost.py).
    """
    raw = _catboost_json(model)
    if "oblivious_trees" not in raw:
        raise ValueError(
            "only CatBoost models with oblivious (symmetric) trees can be "
            "converted; this model has no 'oblivious_trees' section"
        )
    fi = raw.get("features_info", {}) or {}
    for unsupported in ("categorical_features", "ctr_features", "ctrs", "text_features", "embedding_features"):
        if fi.get(unsupported):
            raise ValueError(
                f"CatBoost models using {unsupported} are not supported by from_catboost() (numeric features only)"
            )
    float_features = fi.get("float_features", []) or []
    # NaN routing is a per-feature property in CatBoost, not per-split
    nan_left: Dict[int, bool] = {}
    for f in float_features:
        treatment = str(f.get("nan_value_treatment", "AsIs"))
        # AsFalse sends NaN to the 0 bit (left); AsTrue to the 1 bit (right)
        nan_left[int(f["flat_feature_index"])] = treatment != "AsTrue"
    num_feature = 1 + max([int(f["flat_feature_index"]) for f in float_features] or [-1])

    objective, _ = _catboost_objective(raw)
    scale_and_bias = raw.get("scale_and_bias") or [1.0, [0.0]]
    scale = float(scale_and_bias[0])
    bias_raw = scale_and_bias[1]
    bias_list = list(bias_raw) if isinstance(bias_raw, (list, tuple)) else [bias_raw]
    if len(bias_list) != 1:
        raise ValueError("multiclass CatBoost models are not supported yet")
    bias = float(bias_list[0])

    trees = raw["oblivious_trees"]
    bodies: List[str] = []
    if bias != 0.0:
        bodies.append(_constant_tree(bias))
    for t in trees:
        splits = list(t.get("splits") or [])
        if not splits:
            # a constant tree: CatBoost can emit depth-0 trees
            vals = t["leaf_values"]
            bodies.append(_constant_tree(float(vals[0]) * scale))
            continue
        for sp in splits:
            if str(sp.get("split_type", "FloatFeature")) != "FloatFeature":
                raise ValueError(
                    f"CatBoost split_type '{sp.get('split_type')}' is not supported (numeric FloatFeature splits only)"
                )
            sp["nan_left"] = nan_left.get(int(sp["float_feature_index"]), True)
        leaf_values = t["leaf_values"]
        if len(leaf_values) != (1 << len(splits)):
            raise ValueError(
                f"expected {1 << len(splits)} leaves for a depth-{len(splits)} "
                f"oblivious tree, got {len(leaf_values)} -- multiclass CatBoost "
                "models are not supported yet"
            )
        bodies.append(_oblivious_to_text(splits, leaf_values, t.get("leaf_weights"), scale))

    if feature_names is None:
        names = [str(f.get("feature_id") or "") for f in float_features]
        feature_names = (
            names if all(names) and len(names) == num_feature else [f"Column_{i}" for i in range(num_feature)]
        )
    feature_names = [str(n).replace(" ", "_") for n in feature_names]

    chunks = [f"Tree={i}\n{body}\n" for i, body in enumerate(bodies)]
    header = (
        "tree\n"
        "version=v4\n"
        "num_class=1\n"
        "num_tree_per_iteration=1\n"
        "label_index=0\n"
        f"max_feature_idx={num_feature - 1}\n"
        f"objective={objective}\n"
        f"feature_names={' '.join(feature_names)}\n"
        f"feature_infos={' '.join(['none'] * num_feature)}\n"
        f"tree_sizes={' '.join(str(len(c)) for c in chunks)}\n\n"
    )
    return Booster(model_str=header + "".join(chunks) + "end of trees\n")
