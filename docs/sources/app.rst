.. _app:

Using the VERTEX App
====================

The VERTEX app can be used online at https://vertex.isaric.org, or you can build your own local version and run it (via Docker) using the following instructions.


.. docker:

Running the App
---------------

As mentioned in the :doc:`Getting Started guide <getting-started>` the app can be run locally either as a Python (Plotly Dash) application or in Docker (recommended). The steps below represent the best way to run VERTEX in Docker:

1. Checkout the local VERTEX Git branch on which you want to build and run the app - usually this will be the ``main`` branch, but it could also be any feature or fix branch. If you have access to a command line shell you can do this using:

.. code:: shell

   git checkout <target branch name>

.. note::

   If you have any unsaved/unstaged/uncommited changes on your current branch then you can either stage and commit these (a combination of :command:`git add` and :command:`git commit`), or discard (:command:`git reset`) or stash them somewhere (:command:`git stash`), before switching to the target branch.

2. Build the Docker image (named ``isaric-vertex``) on the branch, from the root of the repository, using:

.. code:: shell

   docker build -t isaric-vertex .

3. Run the app in the container (named ``isaric-vertex``) using the image:

.. code:: shell

   docker run -it \
     --rm -v "$PWD":/app \
     -v "$(pwd)/demo-projects:/app/demo-projects" \
     -v "$(pwd)/projects:/app/projects" \
     -p 8050:8050 \
     -e VERTEX_PROJECTS_DIR="/app/projects" \
     -e VERTEX_ENABLE_SAVE_OUTPUTS="true" \
     -w /app isaric-vertex gunicorn \
     --workers 1 \
     --reload \
     --bind 0.0.0.0:8050 \
     vertex.descriptive_dashboard:server

The app will accessible at ``http://localhost:8050``.

.. note::

   If you're also running a local version of BRIDGE make sure the ports don't conflict - use port ``80`` for BRIDGE and port ``8050`` for VERTEX.

You can also run the app directly (outside of Docker) just using Python, but this requires more precise control over the environment and VERTEX dependencies, as described :ref:`here <requirements>`.

.. _project-structure:

Project Structure
-----------------

The correct project structure is required in order for VERTEX to process, generate and display the analytics and visualisations for the project correctly. Correct project structure depends on two elements:

- a **configuration** file, named :file:`config_file.json` in the root of the project folder
- files for generating the figures and tables for the **insight panels**

Both of these elements can vary depending on the project type, of which there are two kinds:

- **analysis** (or **dynamic**) - the source data is fetched from a project-specific REDCap database (via the REDCap API) and transformed in a suitable format, before the figures and tables are dynamically generated from Python libraries
- **prebuilt** (or **static**) - figures and tables are generated from pre-generated data (CSV) and metadata (JSON) files

These variations are described in more detail below.

.. _analysis-projects:

Analysis/Dynamic Projects
~~~~~~~~~~~~~~~~~~~~~~~~~

The key fields in the configuration file (:file:`config_file.json`) for analysis projects include:

- ``"api_url"`` - defines the REDCap API URL for the project database
- ``"api_key"`` - defines the REDCap API key for the project database
- ``"insight_panels_path"`` - defines the relative path of the project subfolder containing the insight panel code

An example is given below of the `config JSON <https://github.com/ISARICResearch/VERTEX/blob/main/demo-projects/ARChetypeCRF_dengue_synthetic/config_file.json>`_ for the `Dengue Synthetic demo analysis project <https://github.com/ISARICResearch/VERTEX/tree/main/demo-projects/ARChetypeCRF_dengue_synthetic>`_:

.. literalinclude:: ../../demo-projects/ARChetypeCRF_dengue_synthetic/config_file.json
   :linenos:

The insight panel files, which for analysis projects must be in the form of Python :file:`.py` files (modules), should be named appropriately in relation to the associated clinical characterisation stages or events, and the files must all be included together in a subfolder, typically named :file:`insight_panels`, that should correspond to the value of the ``"insight_panels_path"`` key in the configuration JSON. A typical organisation is illustrated below for the Dengue Synthetic demo project:

.. code:: shell

   demo-projects/ARChetypeCRF_dengue_synthetic/
   ├── config_file.json
   └── insight_panels
       ├── enrolment_details.py
       ├── outcomes_complications.py
       ├── presentation_demogcomor.py
       ├── presentation_symptoms.py
       └── treatments_interventions.py

   2 directories, 6 files

The structure of the insight panel Python modules is typically in the form given below, here, for example, using the ``enrolment_details`` insight panel in the Dengue Synthetic demo project:

.. literalinclude:: ../../demo-projects/ARChetypeCRF_dengue_synthetic/insight_panels/enrolment_details.py
   :linenos:

For insight panel modules in this format there are two points to note:

- the ``define_button`` function defines the insight panel button in the main dashboard menu by returning a dictionary in the following format:

.. code:: python

   {
       "item": button_item,
       "label": button_label
   }

Every insight panel must define a button item and a button label, and multiple insight panels can share and are grouped by the same button item in the dashboard menu. But the combination of button item and button label must be unique for a given insight panel, to avoid conflicts.

- the ``create_visuals`` function defines the creation of the figures and tables relevant for the insight panel, and must return an (ordered) tuple of Plotly :py:class:`~plotly.graph_objects.Figure` objects as created by the relevant function in the `ISARICAnalytics visualisation <https://isaricanalytics.readthedocs.io/en/latest/sources/isaricanalytics/visualisation.html#module-isaricanalytics.visualisation>`_  library.


An alternative, slightly simpler format for the insight panel modules is given below from the ``enrolment_details`` insight panel of a local project variant of the same Dengue Synthetic demo project:

.. code:: python

    import json
    from pathlib import Path

    import isaricanalytics.visualisation as idw
    import pandas as pd

    CONFIG_DICT = json.loads(Path(__file__).parent.parent.joinpath("config_file.json").read_text())
    RESEARCH_QUESTION_ITEM = "Enrolment"
    RESEARCH_QUESTION_ITEM_LABEL = "Enrolment Details"


    def main(df_map, df_forms_dict, dictionary):
        """
        Create all visuals in the insight panel from the RAP dataframe
        """
        df_sunburst = df_map[["subjid", "site", "filters_country"]].groupby(["site", "filters_country"]).nunique().reset_index()
        df_sunburst["site"] = df_sunburst["site"].str.split("-", n=1).str[0]

        fig_patients_bysite = idw.fig_sunburst(
            df_sunburst,
            title="Enrolment by site",
            path=["filters_country", "site"],
            values="subjid",
            suffix=__name__,
            filepath=Path(__file__).parent.parent.joinpath(CONFIG_DICT["outputs_path"]),
            save_inputs=CONFIG_DICT["save_outputs"],
            graph_label="Site Enrolment*",
            graph_about="...",
        )

        disclaimer_text = """Disclaimer: the underlying data for these figures is \
    synthetic data. Results may not be clinically relevant or accurate."""
        disclaimer_df = pd.DataFrame(disclaimer_text, columns=["paragraphs"], index=range(1))
        disclaimer = idw.fig_text(
            disclaimer_df,
            suffix=__name__,
            filepath=Path(__file__).parent.parent.joinpath(CONFIG_DICT["outputs_path"]),
            save_inputs=CONFIG_DICT["save_outputs"],
            graph_label="*DISCLAIMER: SYNTHETIC DATA*",
            graph_about=disclaimer_text,
        )

        return (fig_patients_bysite, disclaimer)

Here, the insight panel button item and label are defined not via a function but by global attributes (``RESEARCH_QUESTION_ITEM`` and ``RESEARCH_QUESTION_ITEM_LABEL``), and the ``main`` function replaces the ``create_visuals`` function but has the same shape and return value as in the example above. Note here also that the configuration JSON is loaded into the ``CONFIG_DICT`` variable to be available to the ``main`` function for use.

.. _static-projects:

Prebuilt/Static Projects
~~~~~~~~~~~~~~~~~~~~~~~~

The static projects configuration JSON file can omit the REDCap API fields entirely (or, alternatively, leave them blank) and instead define two additional fields for the VERTEX global map in the dashboard view and generating the figures and tables for the insight panels:

- :file:`dashboard_data.csv` - defines the top-level country and patient count data for the affected country (or countries), e.g. for the Plague Bubo Images project:

.. code:: csv

   country_iso,country_name,country_count
   MDG,Madagascar,20

- :file:`dashboard_metadata.json` - defines the static figure and table data and metadata, e.g. for the Plague Bubo Images project:

.. code:: json

    {
        "insight_panels": [
            {
                "item": "Bubo Clinical Details",
                "label": "Clinical Presentation",
                "suffix": "clinical_presentation",
                "graph_ids": [
                    "clinical_presentation/fig_table",
                    "clinical_presentation/fig_table_OtherCharacteristics",
                    "clinical_presentation/fig_dual_stack_pyramid"
                ]
            },
            {
                "item": "Bubo Clinical Details",
                "label": "Bubo Sonographic Characteristics",
                "suffix": "sonographics_characteristics",
                "graph_ids": [
                    "sonographics_characteristics/fig_table_ImageDescription",
                    "sonographics_characteristics/fig_table_SonographicCharacteristics",
                    "sonographics_characteristics/fig_dual_stack_pyramid"
                ]
            },
            {
                "item": "Bubo Clinical Details",
                "label": "Characteristics",
                "suffix": "characteristics",
                "graph_ids": [
                    "characteristics/fig_upset_CharacteristicsConfirme_day1",
                    "characteristics/fig_upset_CharacteristicsConfirme_day11",
                    "characteristics/fig_upset_CharacteristicsNonCase_day1",
                    "characteristics/fig_upset_CharacteristicsNonCase_day11"
                ]
            },
            {
                "item": "Bubo Clinical Details",
                "label": "Reviewers Comparison",
                "suffix": "reviewers_comparison",
                "graph_ids": [
                    "reviewers_comparison/fig_heatmaps_zone",
                    "reviewers_comparison/fig_heatmaps_day"
                ]
            }
        ]
    }

As insight panel figures and tables for static projects are generated from static data (CSV) and metadata (JSON) files no insight panel Python files are required or used. Instead, for each figure or table a data CSV and a metadata JSON file are required, and all these files should be organised according to some folder structure. This can be discussed with the :email:`ISARIC data team <data@isaric.org>` on request.
