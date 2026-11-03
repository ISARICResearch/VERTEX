"""Deprecated compatibility shim for legacy ``vertex.IsaricDraw`` imports."""

import warnings as _warnings

_warnings.warn(
    "vertex.IsaricDraw is deprecated; import from isaricanalytics.visualisation instead.",
    DeprecationWarning,
    stacklevel=2,
)

from isaricanalytics.visualisation import *  # noqa: F401,F403,E402
