import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("legacy_module", "target_module", "exported_name"),
    [
        ("vertex.IsaricAnalytics", "IsaricAnalytics", "descriptive_table"),
        ("vertex.IsaricDraw", "IsaricDraw", "fig_text"),
        ("vertex.getREDCapData", "getREDCapData", "get_records"),
    ],
)
def test_legacy_vertex_module_shims_warn_and_reexport(monkeypatch, legacy_module, target_module, exported_name):
    target = importlib.import_module(f"isaricanalytics.{target_module}")
    monkeypatch.delitem(sys.modules, legacy_module, raising=False)

    namespace = {}
    with pytest.warns(DeprecationWarning, match=legacy_module):
        exec(f"from {legacy_module} import *", namespace, namespace)

    assert namespace[exported_name] is getattr(target, exported_name)
