"""Deprecated compatibility shim for legacy ``vertex.getREDCapData`` imports."""

import warnings as _warnings

_warnings.warn(
    "vertex.getREDCapData is deprecated; import from isaricanalytics.getREDCapData instead.",
    DeprecationWarning,
    stacklevel=2,
)

from isaricanalytics.getREDCapData import *  # noqa: F401,F403,E402
