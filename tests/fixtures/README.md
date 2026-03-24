# Test Fixture Projects

This folder contains minimal, committed project fixtures used by tests.
Tests should copy these fixtures into `tmp_path` and mutate only the files
required for the specific edge case under test.

## Fixture: `analysis_files_project`

Purpose:
- Minimal file-backed analysis project for `load_vertex_from_files` and config
  loading tests.

Required structure:

```text
analysis_files_project/
  config_file.json
  insight_panels/
    presentation.py
  analysis_data/
    vertex_dictionary.csv
    df_map.csv
    presentation.csv
    invalid_no_subjid.csv
```

Schema notes:
- `config_file.json` includes:
  - `project_name`, `project_id`, `project_owner`, `is_public`
  - `insight_panels_path`
  - `insight_panels_data_path`
- `analysis_data/vertex_dictionary.csv` must include `field_name`, `field_type`,
  and `field_label`.
- `analysis_data/df_map.csv` should include `subjid` and any date variables
  referenced in the dictionary.
- Form CSVs intended to be loaded into `df_forms_dict` must include `subjid`.

## Fixture: `prebuilt_public_project`

Purpose:
- Minimal static/public-output project for `load_public_dashboard`,
  `get_public_visuals`, and prebuilt `load_project_data` tests.

Required structure:

```text
prebuilt_public_project/
  config_file.json
  dashboard_metadata.json
  dashboard_data.csv
  panel_a/
    fig_text_metadata.json
    fig_text_data___0.csv
```

Schema notes:
- `config_file.json` includes:
  - `project_name`, `project_id`, `project_owner`, `is_public`
- `dashboard_metadata.json` includes:
  - `insight_panels`: list of button objects with at least `suffix`
- Figure metadata files (for example `fig_text_metadata.json`) include:
  - `fig_id`
  - `fig_name` (must map to a function in `vertex.IsaricDraw`)
  - `fig_arguments` (kwargs for the draw function)
  - `fig_data` (list of CSV paths relative to project root)
- `dashboard_data.csv` provides prebuilt country counts with:
  - `country_iso`, `country_name`, `country_count`
