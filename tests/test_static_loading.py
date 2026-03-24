from vertex.io import load_public_dashboard
from vertex.layout.insight_panels import get_public_visuals


def test_load_public_dashboard_reads_default_file(prebuilt_project_factory):
    project_dir = prebuilt_project_factory()

    metadata = load_public_dashboard(str(project_dir), {})

    assert "insight_panels" in metadata
    assert metadata["insight_panels"][0]["suffix"] == "panel_a"


def test_load_public_dashboard_honors_custom_dashboard_metadata_filename(prebuilt_project_factory):
    project_dir = prebuilt_project_factory(metadata_filename="custom_meta.json")

    metadata = load_public_dashboard(str(project_dir), {"dashboard_metadata": "custom_meta.json"})

    assert metadata["insight_panels"][0]["suffix"] == "panel_a"


def test_load_public_dashboard_missing_metadata_should_not_crash(prebuilt_project_factory):
    project_dir = prebuilt_project_factory()
    (project_dir / "dashboard_metadata.json").unlink()

    metadata = load_public_dashboard(str(project_dir), {})

    # Desired fallback once hardened:
    assert metadata == {"insight_panels": []}


def test_get_public_visuals_builds_visual_for_valid_fig_text(prebuilt_project_factory):
    project_dir = prebuilt_project_factory(
        buttons=[{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text"], "title": "Panel A"}]
    )
    buttons = [{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text"], "title": "Panel A"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), buttons)
    visuals = visuals_by_suffix["panel_a"].create_visuals()

    assert "panel_a" in visuals_by_suffix
    assert len(visuals) == 1
    _, fig_id, graph_label, _ = visuals[0]
    assert fig_id == "panel_a/fig_text"
    assert graph_label == "Test graph"


def test_get_public_visuals_returns_empty_panel_when_suffix_dir_missing(prebuilt_project_factory):
    project_dir = prebuilt_project_factory(
        buttons=[{"suffix": "missing_suffix", "graph_ids": ["missing_suffix/fig_text"], "title": "Missing"}],
        panel_metadata_files={},
        data_files={},
        create_panel_dir=False,
    )
    buttons = [{"suffix": "missing_suffix", "graph_ids": ["missing_suffix/fig_text"], "title": "Missing"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), buttons)

    assert visuals_by_suffix["missing_suffix"].create_visuals() == []


def test_get_public_visuals_skips_missing_csv_and_continues(prebuilt_project_factory):
    project_dir = prebuilt_project_factory(
        panel_metadata_files={
            "fig_text_metadata.json": {
                "fig_id": "panel_a/fig_text",
                "fig_name": "fig_text",
                "fig_arguments": {"suffix": "panel_a", "graph_id": "fig_text"},
                "fig_data": ["panel_a/missing_input.csv"],
            }
        },
        data_files={},
    )
    buttons = [{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text"], "title": "Panel A"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), buttons)

    assert visuals_by_suffix["panel_a"].create_visuals() == []


def test_get_public_visuals_skips_unknown_draw_function(prebuilt_project_factory):
    project_dir = prebuilt_project_factory(
        panel_metadata_files={
            "fig_unknown_metadata.json": {
                "fig_id": "panel_a/fig_unknown",
                "fig_name": "fig_not_a_real_draw_function",
                "fig_arguments": {"suffix": "panel_a", "graph_id": "fig_unknown"},
                "fig_data": ["panel_a/fig_unknown_data___0.csv"],
            }
        },
        data_files={"panel_a/fig_unknown_data___0.csv": "paragraphs\nunknown\n"},
    )
    buttons = [{"suffix": "panel_a", "graph_ids": ["panel_a/fig_unknown"], "title": "Panel A"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), buttons)

    assert visuals_by_suffix["panel_a"].create_visuals() == []


def test_get_public_visuals_skips_malformed_metadata_json(prebuilt_project_factory):
    project_dir = prebuilt_project_factory()
    (project_dir / "panel_a" / "fig_text_metadata.json").write_text("{ this is not valid json")
    buttons = [{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text"], "title": "Panel A"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), buttons)

    assert visuals_by_suffix["panel_a"].create_visuals() == []


def test_get_public_visuals_handles_non_list_fig_data(prebuilt_project_factory):
    project_dir = prebuilt_project_factory(
        panel_metadata_files={
            "fig_text_metadata.json": {
                "fig_id": "panel_a/fig_text",
                "fig_name": "fig_text",
                "fig_arguments": {"suffix": "panel_a", "graph_id": "fig_text"},
                # malformed type: expected list
                "fig_data": "panel_a/fig_text_data___0.csv",
            }
        }
    )
    buttons = [{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text"], "title": "Panel A"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), buttons)

    assert visuals_by_suffix["panel_a"].create_visuals() == []


def test_get_public_visuals_button_missing_suffix_should_not_crash(prebuilt_project_factory):
    project_dir = prebuilt_project_factory()
    malformed_buttons = [{"graph_ids": ["panel_a/fig_text"], "title": "Panel A"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), malformed_buttons)

    # Desired fallback once hardened:
    assert visuals_by_suffix == {}


def test_get_public_visuals_filters_metadata_by_graph_ids(prebuilt_project_factory):
    project_dir = prebuilt_project_factory(
        buttons=[{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text_target"], "title": "Panel A"}],
        panel_metadata_files={
            "fig_text_target_metadata.json": {
                "fig_id": "panel_a/fig_text_target",
                "fig_name": "fig_text",
                "fig_arguments": {"suffix": "panel_a", "graph_id": "fig_text_target"},
                "fig_data": ["panel_a/fig_text_target_data___0.csv"],
            },
            "fig_alt_metadata.json": {
                "fig_id": "panel_a/fig_alt",
                "fig_name": "fig_text",
                "fig_arguments": {"suffix": "panel_a", "graph_id": "fig_alt"},
                "fig_data": ["panel_a/fig_alt_data___0.csv"],
            },
        },
        data_files={
            "panel_a/fig_text_target_data___0.csv": "paragraphs\ntarget\n",
            "panel_a/fig_alt_data___0.csv": "paragraphs\nalt\n",
        },
    )
    buttons = [{"suffix": "panel_a", "graph_ids": ["panel_a/fig_text_target"], "title": "Panel A"}]

    visuals_by_suffix, _ = get_public_visuals(str(project_dir), buttons)
    visuals = visuals_by_suffix["panel_a"].create_visuals()

    assert len(visuals) == 1
    assert visuals[0][1] == "panel_a/fig_text_target"
