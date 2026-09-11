"""Register bundled Qt DLL directories before PyQt6 is imported."""

import os
import sys

_HANDLES = []
_PRELOADED = []

# Resolve dependencies only from the loaded DLL's directory, directories
# registered below, and System32. In particular, do not allow an unrelated Qt
# installation beside the onefile executable to override the bundled runtime.
_SAFE_LOAD_FLAGS = 0x00000100 | 0x00000400 | 0x00000800


def configure_qt_dll_search_path():
    if sys.platform != "win32" or not hasattr(sys, "_MEIPASS"):
        return

    base_dir = sys._MEIPASS
    candidates = [
        os.path.join(base_dir, "PyQt6", "Qt6", "bin"),
        os.path.join(base_dir, "PyQt6", "Qt", "bin"),
        base_dir,
    ]
    search_dirs = [path for path in candidates if os.path.isdir(path)]

    if hasattr(os, "add_dll_directory"):
        for directory in search_dirs:
            _HANDLES.append(os.add_dll_directory(directory))

    path_parts = [
        part for part in os.environ.get("PATH", "").split(os.pathsep) if part
    ]
    existing = {os.path.normcase(os.path.abspath(part)) for part in path_parts}
    extra_dirs = [
        directory
        for directory in search_dirs
        if os.path.normcase(os.path.abspath(directory)) not in existing
    ]
    if extra_dirs:
        os.environ["PATH"] = os.pathsep.join(extra_dirs + path_parts)

    import ctypes

    preload_paths = [
        os.path.join(search_dirs[0], "Qt6Core.dll") if search_dirs else "",
    ]
    for path in preload_paths:
        if path and os.path.isfile(path):
            _PRELOADED.append(ctypes.WinDLL(path, winmode=_SAFE_LOAD_FLAGS))


configure_qt_dll_search_path()
