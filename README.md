# VERTEX
**ISARIC VERTEX** is a web-based application designed for local use by users. It serves as an analysis tool for data captured through our complementary tools: [ISARIC ARC](https://github.com/ISARICResearch/ARC) and [ISARIC BRIDGE](https://github.com/ISARICResearch/BRIDGE).

VERTEX is a web-based application designed to present graphs and tables based on key relevant research questions that need to be quickly answered during an outbreak. Currently, VERTEX performs descriptive analysis, which can identify the spectrum of clinical features in a disease outbreak. New research questions will be added by the ISARIC team and the wider scientific community, enabling the creation and sharing of additional analysis methods.

## About VERTEX

**ISARIC VERTEX** enables users to connect with a REDCap database through an API call. For detailed instructions, please refer to our [Getting Started with VERTEX guide](https://isaricresearch.github.io/Training/vertex_starting.html).

VERTEX has three main elements:
  - **Main App**: A map that visually represents the number and country of patients in the REDCap database.
  - **Menu**: A menu containing a series of buttons that open different insight panels.
  - **Insight Panels**: Sets of visuals, each related to specific research questions.

VERTEX processes and visualizes data using the concept of **Reproducible Analytical Pipelines (RAPs)**. RAPs are a set of resuable functions or blocks of code that can request specific variables from an [ISARIC ARC](https://github.com/ISARICResearch/ARC)-formatted REDCap database. These functions then process the data to generate dataframes, which can then be visualized interactively through a Plotly Dash app.

## VERTEX Version 1.0

**VERTEX Version 1.0** includes insight panels developed for the following research questions:
- Clinical characterization of on presentation, including:
     - demographics and comorbidities
     - pregnancy
     - transmission or exposure
     - signs, symptoms, labs and vitals from the first 24hr after admission
- Patient outcomes, including complications

Additionally, if you want to create your own insight panel, please follow our [Creating an Insight Panel guide](https://isaricresearch.github.io/Training/insight_panel.html).


## How to Use VERTEX

To get started with VERTEX, please refer to our [Getting Started with VERTEX guide](https://isaricresearch.github.io/Training/vertex_starting.html).

## Testing and Coverage

Run tests locally:

```bash
pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_cov --cov=vertex --cov-report=term-missing --cov-report=xml
```

GitHub Actions uploads `coverage.xml` to Codecov via `.github/workflows/test-and-coverage.yml`.
If the repository is private, set the `CODECOV_TOKEN` repository secret.
The workflow also uploads `coverage.xml` as a GitHub Actions artifact and treats
Codecov upload as best-effort, so test CI still passes if Codecov is unavailable.

Validate static output metadata/schema locally:

```bash
python .github/actions/validate-project-outputs/validate_project_outputs.py --root demo-projects --require-project-files true
```

Reusable CI workflow for schema checks:

- `.github/workflows/validate-project-outputs.yml` can be called from other repositories to validate dashboard metadata and figure metadata/data references before deployment.

## Project Sources

VERTEX loads projects from two roots:

- `demo-projects/`: dynamic analysis projects (API/data-backed, filterable)
- `projects/`: prebuilt static projects (non-analysis mode)

Both paths are configurable with:

- `VERTEX_PROJECTS_DIR`

For prebuilt/static projects, `config_file.json` should include:

- `project_name`
- `project_id`
- `project_owner`
- `is_public`

Current temporary visibility behavior:

- Not logged in: only prebuilt projects with `is_public: true` are shown
- Logged in: all prebuilt projects are shown
- Demo analysis projects are always shown

If you encounter any issues or have suggestions for improvements, we encourage you to submit an [issue](https://github.com/ISARICResearch/VERTEX/issues) on this repository or reach out to us via email at [data@isaric.org](mailto:data@isaric.org).

## Project Entrypoints

Command line entrypoints (executables) are provided for performing specific meaningful actions within the context of the project. The entrypoints are defined in the [project TOML](https://github.com/ISARICResearch/VERTEX/blob/main/pyproject.toml) (in the `[project.scripts]` section) and only become available once the project package (`isaric-vertex`) is installed in editable mode using
```shell
pip install -e .
```
Any changes to the entrypoint definition or function implementing the entrypoint logic requires a re-installation (`pip uninstall -y isaric-vertex` and `pip install -e .`). The entrypoints currently include:

* `descriptive-analytics` - generates analytics outputs from a given project source folder and saving them to local folder (within the project folder), e.g. for the Dengue synthetic demo project
```shell
$ descriptive-analytics demo-projects/ARChetypeCRF_dengue_synthetic/
2026-03-20 12:16:23 [INFO] vertex.descriptive_analytics: Loading project data from project path: "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_dengue_synthetic"
2026-03-20 12:16:25 [INFO] vertex.io: Retrieving data from redcap API
2026-03-20 12:16:26 [INFO] vertex.getREDCapData: REDCap data pipeline start
2026-03-20 12:16:26 [INFO] vertex.getREDCapData: REDCap records export: requesting all records
2026-03-20 12:16:30 [INFO] vertex.getREDCapData: REDCap records export complete in 3.9s (rows=6495)
2026-03-20 12:16:30 [INFO] vertex.getREDCapData: REDCap step get_records finished in 3.9s
2026-03-20 12:16:30 [INFO] vertex.getREDCapData: REDCap step get_data_dictionary finished in 0.2s
2026-03-20 12:16:30 [INFO] vertex.getREDCapData: REDCap step get_missing_data_codes finished in 0.2s
2026-03-20 12:16:32 [INFO] vertex.getREDCapData: REDCap step initial_data_processing finished in 2.3s (rows=6495, cols=685)
2026-03-20 12:16:33 [INFO] vertex.getREDCapData: REDCap step get_form_event finished in 0.5s
2026-03-20 12:16:33 [INFO] vertex.getREDCapData: REDCap step get_df_map finished in 0.1s (rows=1000)
2026-03-20 12:16:33 [INFO] vertex.getREDCapData: REDCap step get_df_forms finished in 0.0s (forms=4)
2026-03-20 12:16:33 [INFO] vertex.getREDCapData: REDCap data pipeline complete in 7.3s
2026-03-20 12:16:33 [INFO] vertex.descriptive_analytics: Saving outputs to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_dengue_synthetic/outputs"
2026-03-20 12:16:33 [WARNING] vertex.io: Folder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_dengue_synthetic/outputs/" already exists, removing this
2026-03-20 12:16:33 [INFO] vertex.io: Saving files for static dashboard to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_dengue_synthetic/outputs/"
2026-03-20 12:16:43 [INFO] vertex.io: Public dashboard files saved to /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_dengue_synthetic/outputs/
```

## Contributors

- Esteban Garcia-Gallo - [esteban.garcia@ndm.ox.ac.uk](mailto:esteban.garcia@ndm.ox.ac.uk)
- Tom Edinburgh - [tom.edinburgh@ndm.ox.ac.uk](mailto:tom.edinburgh@ndm.ox.ac.uk)
- Leonardo Bastos - [lslbastos@puc-rio.br](mailto:lslbastos@puc-rio.br)
- Igor Peres - [igor.peres@puc-rio.br](mailto:igor.peres@puc-rio.br)
- Luiz Eduardo Raffaini [lemraffaini@gmail.com](mailto:lemraffaini@gmail.com)
- Sara Duque-Vallejo - [sara.duquevallejo@ndm.ox.ac.uk](mailto:sara.duquevallejo@ndm.ox.ac.uk)
- Laura Merson - [laura.merson@ndm.ox.ac.uk](mailto:laura.merson@ndm.ox.ac.uk)
- Elise Pesonel - [elise.pesonel@ndm.ox.ac.uk](mailto:elise.pesonel@ndm.ox.ac.uk)
- Sandeep Murthy - [sandeep.murthy@ndm.ox.ac.uk](mailto:sandeep.murthy@ndm.ox.ac.uk)

---

**Note**: VERTEX is maintained by ISARIC. For inquiries, support, or collaboration, please [contact us](mailto:data@isaric.org).
