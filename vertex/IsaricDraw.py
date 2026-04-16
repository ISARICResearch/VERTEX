"""Deprecated compatibility shim for legacy ``vertex.IsaricDraw`` imports."""

import warnings as _warnings

_warnings.warn(
    "vertex.IsaricDraw is deprecated; import from isaricanalytics.IsaricDraw instead.",
    DeprecationWarning,
    stacklevel=2,
)

from isaricanalytics.IsaricDraw import *  # noqa: F401,F403,E402
