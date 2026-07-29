# coding: utf-8
"""Backwards-compatibility shim: ``import lightgbm`` -> Falcata.

Falcata's Python package is ``falcata``. This shim keeps code written
against the pre-rename package name working unchanged::

    import lightgbm as lgb          # still works, gives you Falcata
    from lightgbm import Booster    # ditto

Everything is re-exported from :mod:`falcata`, and the submodules are
registered under both names so ``import lightgbm.basic`` also resolves.

This exists purely for compatibility; new code should ``import falcata``.
"""

import sys as _sys
import warnings as _warnings

import falcata as _falcata
from falcata import *  # noqa: F401,F403

# submodules under the legacy name, so `import lightgbm.basic` etc. resolve
for _name in ("basic", "callback", "compat", "engine", "libpath", "plotting", "sklearn", "dask"):
    _mod = getattr(_falcata, _name, None)
    if _mod is None:
        _mod = _sys.modules.get(f"falcata.{_name}")
    if _mod is not None:
        _sys.modules[f"lightgbm.{_name}"] = _mod
        globals()[_name] = _mod
del _name, _mod

__version__ = _falcata.__version__
__all__ = list(getattr(_falcata, "__all__", []))

if not _sys.warnoptions:
    _warnings.warn(
        "The 'lightgbm' import name is a compatibility alias for Falcata and may be "
        "removed in a future release; use 'import falcata' instead.",
        FutureWarning,
        stacklevel=2,
    )
