"""Deprecated compatibility shim for legacy ``vertex.IsaricAnalytics`` imports."""

import warnings as _warnings

_warnings.warn(
    "vertex.IsaricAnalytics is deprecated; import from isaricanalytics.IsaricAnalytics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from isaricanalytics.IsaricAnalytics import *  # noqa: F401,F403,E402
