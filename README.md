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


## Contributors

- Esteban Garcia-Gallo - [esteban.garcia@ndm.ox.ac.uk](mailto:esteban.garcia@ndm.ox.ac.uk)
- Tom Edinburgh - [tom.edinburgh@ndm.ox.ac.uk](mailto:tom.edinburgh@ndm.ox.ac.uk)
- Leonardo Bastos - [lslbastos@puc-rio.br](mailto:lslbastos@puc-rio.br)
- Igor Peres - [igor.peres@puc-rio.br](mailto:igor.peres@puc-rio.br)
- Luiz Eduardo Raffaini [lemraffaini@gmail.com](mailto:lemraffaini@gmail.com)
- Sara Duque-Vallejo - [sara.duquevallejo@ndm.ox.ac.uk](mailto:sara.duquevallejo@ndm.ox.ac.uk)
- Laura Merson - [laura.merson@ndm.ox.ac.uk](mailto:laura.merson@ndm.ox.ac.uk)
- Elise Pesonel - [elise.pesonel@ndm.ox.ac.uk](mailto:elise.pesonel@ndm.ox.ac.uk)

---

**Note**: VERTEX is maintained by ISARIC. For inquiries, support, or collaboration, please [contact us](mailto:data@isaric.org).
