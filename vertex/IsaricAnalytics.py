"""Deprecated compatibility shim for legacy ``vertex.IsaricAnalytics`` imports."""

import warnings as _warnings

_warnings.warn(
    "vertex.IsaricAnalytics is deprecated; import from isaricanalytics.analytics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from isaricanalytics.analytics import *  # noqa: F401,F403,E402
