# coding: utf-8
"""Find the path to Falcata dynamic library files."""

import ctypes
import os
from os import environ
from pathlib import Path
from platform import system
from typing import List

__all__: List[str] = []

# Keep the handles returned by os.add_dll_directory alive for the life of the
# process; dropping them would remove the directory from the DLL search path.
_DLL_DIR_HANDLES: list = []


def _add_cuda_dll_dirs() -> None:
    """Make the CUDA runtime DLLs discoverable before loading lib_falcata.dll.

    A CUDA-enabled ``lib_falcata.dll`` depends on the CUDA runtime, NVRTC and
    the CUDA driver. Since Python 3.8, ctypes no longer searches ``%PATH%`` for
    a DLL's dependencies on Windows, so loading a CUDA build fails with
    "Could not find module ... (or one of its dependencies)" unless the CUDA
    ``bin`` directory is registered explicitly via ``os.add_dll_directory``.

    This is Windows-only and a no-op for CPU-only builds (which have no CUDA
    dependency) -- registering the directory is harmless if it is never used.
    """
    if system() not in ("Windows", "Microsoft"):
        return
    candidates = []
    cuda_path = environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    # also honor any CUDA 'bin' directory already present on PATH
    for entry in environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        p = Path(entry)
        if p.name.lower() == "bin" and "cuda" in str(p).lower():
            candidates.append(p)
    seen = set()
    for directory in candidates:
        key = str(directory).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if directory.is_dir():
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))  # type: ignore[attr-defined]
        except OSError:
            # a non-existent or inaccessible directory is not fatal: the load
            # below will surface a clear error if a dependency is truly missing
            pass


def _find_lib_path() -> List[str]:
    """Find the path to Falcata library files.

    Returns
    -------
    lib_path: list of str
       List of all found library paths to Falcata.
    """
    curr_path = Path(__file__).resolve()
    dll_path = [
        curr_path.parents[1],
        curr_path.parents[0] / "bin",
        curr_path.parents[0] / "lib",
    ]
    if system() in ("Windows", "Microsoft"):
        dll_path.append(curr_path.parents[1] / "Release")
        dll_path.append(curr_path.parents[1] / "windows" / "x64" / "DLL")
        dll_path = [p / "lib_falcata.dll" for p in dll_path]
    elif system() == "Darwin":
        dll_path = [p / "lib_falcata.dylib" for p in dll_path]
    else:
        dll_path = [p / "lib_falcata.so" for p in dll_path]
    lib_path = [str(p) for p in dll_path if p.is_file()]
    if not lib_path:
        dll_path_joined = "\n".join(map(str, dll_path))
        raise Exception(f"Cannot find falcata library file in following paths:\n{dll_path_joined}")
    return lib_path


# we don't need lib_falcata while building docs
_LIB: ctypes.CDLL
if environ.get("FALCATA_BUILD_DOC", "False") == "True":
    from unittest.mock import Mock  # isort: skip

    _LIB = Mock(ctypes.CDLL)  # type: ignore
else:
    _add_cuda_dll_dirs()
    _LIB = ctypes.cdll.LoadLibrary(_find_lib_path()[0])
