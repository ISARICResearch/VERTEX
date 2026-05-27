.. _cli:

Command Line Interface (CLI)
============================

VERTEX provides a very simple command line interface (CLI) currently consisting of a single project entrypoint (executable or console script) named :program:`descriptive-analytics`, which becomes available once the project is installed in `editable mode <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_:

.. code:: shell

   pip install -e .

This will install the project locally in a package named ``isaric-vertex``, and set up the executable.

.. _cli-descriptive-analytics:

Descriptive Analytics
---------------------

Currently, there is a single executable, :program:`descriptive-analytics`, which exports all figures and tables from a project's insight panels, given the project path, to a subfolder named :file:`output` in the project folder:

.. code:: shell

   $ descriptive-analytics --help

   ...

   Options:
     --project-path TEXT  The (absolute or relative) path to the project.
                          [required]
     --help               Show this message and exit.

The project path can be either a relative path (relative to the working directory) or an absolute path, and the project itself can either be an "analysis" project where the data is fetched from an associated REDCap project database via the REDCap API and figures and artifacts are dynamically generated, or a "static" project with pre-generated aggregated figure and artifact metadata files.

An example run is given below for the **ARChetypeCRF_mpox_synthetic** demo MPox analysis project with synthetic data:

.. code:: shell

   $ descriptive-analytics --project-path demo-projects/ARChetypeCRF_mpox_synthetic/
   2026-05-26 16:25:24 [INFO] vertex.descriptive_analytics: Loading project data from project path: "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic"
   2026-05-26 16:25:36 [WARNING] vertex.layout.insight_panels: The `define_button` function will not be supported in future VERTEX releases. Please use `RESEARCH_QUESTION_ITEM` and `RESEARCH_QUESTION_ITEM_LABEL` attributes to define the button instead.
   2026-05-26 16:25:36 [INFO] vertex.io: Loading data from /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic
   2026-05-26 16:25:38 [INFO] vertex.descriptive_analytics: Saving public outputs to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs"
   2026-05-26 16:25:38 [WARNING] vertex.io: Folder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/" already exists, removing this
   2026-05-26 16:25:38 [INFO] vertex.io: Saving files for static dashboard to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/"
   2026-05-26 16:25:38 [WARNING] vertex.layout.insight_panels: The `create_visuals` function will not be supported in future VERTEX releases. Please use a `main` function instead.
   2026-05-26 16:25:50 [INFO] vertex.io: Public dashboard files saved to /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/
   2026-05-26 16:25:50 [INFO] vertex.descriptive_analytics: Saving all figures to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs"
   2026-05-26 16:25:50 [INFO] vertex.io: Saving "enrolment_details" insight panel non-table figures to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/enrolment_details"
   2026-05-26 16:25:54 [INFO] vertex.io: Saved "fig_sunburst.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/enrolment_details/fig_sunburst.png"
   2026-05-26 16:25:55 [INFO] vertex.io: Saved "fig_text.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/enrolment_details/fig_text.png"
   2026-05-26 16:25:55 [INFO] vertex.io: Copying "enrolment_details" table CSVs from CSV output subfolder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/enrolment_details"
   2026-05-26 16:25:55 [INFO] vertex.io: Copying figure table CSVs from "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/enrolment_details" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/enrolment_details"
   2026-05-26 16:25:55 [WARNING] vertex.io: No figure table CSVs found in "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/enrolment_details"
   2026-05-26 16:25:55 [INFO] vertex.io: Cleaning figure table CSVs
   2026-05-26 16:25:55 [INFO] vertex.io: Saving "presentation_demogcomor" insight panel non-table figures to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor"
   2026-05-26 16:25:57 [INFO] vertex.io: Saved "fig_frequency_chart.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor/fig_frequency_chart.png"
   2026-05-26 16:25:59 [INFO] vertex.io: Saved "fig_dual_stack_pyramid.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor/fig_dual_stack_pyramid.png"
   2026-05-26 16:26:01 [INFO] vertex.io: Saved "fig_text.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor/fig_text.png"
   2026-05-26 16:26:03 [INFO] vertex.io: Saved "fig_upset.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor/fig_upset.png"
   2026-05-26 16:26:03 [INFO] vertex.io: Copying "presentation_demogcomor" table CSVs from CSV output subfolder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/presentation_demogcomor"
   2026-05-26 16:26:03 [INFO] vertex.io: Copying figure table CSVs from "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/presentation_demogcomor" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor"
   2026-05-26 16:26:03 [INFO] vertex.io: Copying "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/presentation_demogcomor/fig_table_data___0.csv" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor"
   2026-05-26 16:26:03 [INFO] vertex.io: 1 figure table CSVs copied to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor"
   2026-05-26 16:26:03 [INFO] vertex.io: Cleaning figure table CSVs
   2026-05-26 16:26:03 [INFO] vertex.io: Cleaning figure table CSV /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_demogcomor/fig_table_data___0.csv
   2026-05-26 16:26:03 [INFO] vertex.io: Saving "presentation_symptoms" insight panel non-table figures to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms"
   2026-05-26 16:26:04 [INFO] vertex.io: Saved "fig_frequency_chart.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms/fig_frequency_chart.png"
   2026-05-26 16:26:06 [INFO] vertex.io: Saved "fig_text.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms/fig_text.png"
   2026-05-26 16:26:08 [INFO] vertex.io: Saved "fig_upset.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms/fig_upset.png"
   2026-05-26 16:26:08 [INFO] vertex.io: Copying "presentation_symptoms" table CSVs from CSV output subfolder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/presentation_symptoms"
   2026-05-26 16:26:08 [INFO] vertex.io: Copying figure table CSVs from "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/presentation_symptoms" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms"
   2026-05-26 16:26:08 [INFO] vertex.io: Copying "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/presentation_symptoms/fig_table_data___0.csv" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms"
   2026-05-26 16:26:08 [INFO] vertex.io: 1 figure table CSVs copied to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms"
   2026-05-26 16:26:08 [INFO] vertex.io: Cleaning figure table CSVs
   2026-05-26 16:26:08 [INFO] vertex.io: Cleaning figure table CSV /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/presentation_symptoms/fig_table_data___0.csv
   2026-05-26 16:26:08 [INFO] vertex.io: Saving "lesion_assessment" insight panel non-table figures to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/lesion_assessment"
   2026-05-26 16:26:10 [INFO] vertex.io: Saved "fig_text.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/lesion_assessment/fig_text.png"
   2026-05-26 16:26:10 [INFO] vertex.io: Copying "lesion_assessment" table CSVs from CSV output subfolder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/lesion_assessment"
   2026-05-26 16:26:10 [INFO] vertex.io: Copying figure table CSVs from "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/lesion_assessment" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/lesion_assessment"
   2026-05-26 16:26:10 [INFO] vertex.io: Copying "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/lesion_assessment/fig_table_data___0.csv" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/lesion_assessment"
   2026-05-26 16:26:10 [INFO] vertex.io: 1 figure table CSVs copied to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/lesion_assessment"
   2026-05-26 16:26:10 [INFO] vertex.io: Cleaning figure table CSVs
   2026-05-26 16:26:10 [INFO] vertex.io: Cleaning figure table CSV /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/lesion_assessment/fig_table_data___0.csv
   2026-05-26 16:26:10 [INFO] vertex.io: Saving "treatments_interventions" insight panel non-table figures to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions"
   2026-05-26 16:26:11 [INFO] vertex.io: Saved "fig_frequency_chart.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions/fig_frequency_chart.png"
   2026-05-26 16:26:13 [INFO] vertex.io: Saved "fig_upset.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions/fig_upset.png"
   2026-05-26 16:26:15 [INFO] vertex.io: Saved "fig_upset.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions/fig_upset.png"
   2026-05-26 16:26:17 [INFO] vertex.io: Saved "fig_text.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions/fig_text.png"
   2026-05-26 16:26:19 [INFO] vertex.io: Saved "fig_frequency_chart.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions/fig_frequency_chart.png"
   2026-05-26 16:26:19 [INFO] vertex.io: Copying "treatments_interventions" table CSVs from CSV output subfolder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/treatments_interventions"
   2026-05-26 16:26:19 [INFO] vertex.io: Copying figure table CSVs from "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/treatments_interventions" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions"
   2026-05-26 16:26:19 [INFO] vertex.io: Copying "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/treatments_interventions/fig_table_data___0.csv" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions"
   2026-05-26 16:26:19 [INFO] vertex.io: 1 figure table CSVs copied to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions"
   2026-05-26 16:26:19 [INFO] vertex.io: Cleaning figure table CSVs
   2026-05-26 16:26:19 [INFO] vertex.io: Cleaning figure table CSV /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/treatments_interventions/fig_table_data___0.csv
   2026-05-26 16:26:19 [INFO] vertex.io: Saving "outcomes_complications" insight panel non-table figures to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications"
   2026-05-26 16:26:20 [INFO] vertex.io: Saved "fig_frequency_chart.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications/fig_frequency_chart.png"
   2026-05-26 16:26:22 [INFO] vertex.io: Saved "fig_text.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications/fig_text.png"
   2026-05-26 16:26:24 [INFO] vertex.io: Saved "fig_upset.png" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications/fig_upset.png"
   2026-05-26 16:26:24 [INFO] vertex.io: Copying "outcomes_complications" table CSVs from CSV output subfolder "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/outcomes_complications"
   2026-05-26 16:26:24 [INFO] vertex.io: Copying figure table CSVs from "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/outcomes_complications" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications"
   2026-05-26 16:26:24 [INFO] vertex.io: Copying "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/outcomes_complications/fig_table_data___0.csv" to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications"
   2026-05-26 16:26:24 [INFO] vertex.io: 1 figure table CSVs copied to "/Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications"
   2026-05-26 16:26:24 [INFO] vertex.io: Cleaning figure table CSVs
   2026-05-26 16:26:24 [INFO] vertex.io: Cleaning figure table CSV /Users/smurthy/Documents/srm/dev/VERTEX/demo-projects/ARChetypeCRF_mpox_synthetic/outputs/visuals/outcomes_complications/fig_table_data___0.csv

The :file:`output` subfolder structure can be inspected by using a command line tool such as :program:`tree` (use :command:`brew install tree` on MacOS or :command:`apt install tree` on Linux):

.. code:: shell

   $ tree -L 4 demo-projects/ARChetypeCRF_mpox_synthetic/outputs/
   demo-projects/ARChetypeCRF_mpox_synthetic/outputs/
   ├── config_file.json
   ├── dashboard_data.csv
   ├── dashboard_metadata.json
   ├── enrolment_details
   │   ├── fig_sunburst_data___0.csv
   │   ├── fig_sunburst_metadata.json
   │   ├── fig_text_data___0.csv
   │   └── fig_text_metadata.json
   ├── lesion_assessment
   │   ├── fig_table_data___0.csv
   │   ├── fig_table_metadata.json
   │   ├── fig_text_data___0.csv
   │   └── fig_text_metadata.json
   ├── outcomes_complications
   │   ├── fig_frequency_chart_data___0.csv
   │   ├── fig_frequency_chart_metadata.json
   │   ├── fig_table_data___0.csv
   │   ├── fig_table_metadata.json
   │   ├── fig_text_data___0.csv
   │   ├── fig_text_metadata.json
   │   ├── fig_upset_data___0.csv
   │   ├── fig_upset_data___1.csv
   │   └── fig_upset_metadata.json
   ├── presentation_demogcomor
   │   ├── fig_dual_stack_pyramid_data___0.csv
   │   ├── fig_dual_stack_pyramid_metadata.json
   │   ├── fig_frequency_chart_data___0.csv
   │   ├── fig_frequency_chart_metadata.json
   │   ├── fig_table_data___0.csv
   │   ├── fig_table_metadata.json
   │   ├── fig_text_data___0.csv
   │   ├── fig_text_metadata.json
   │   ├── fig_upset_data___0.csv
   │   ├── fig_upset_data___1.csv
   │   └── fig_upset_metadata.json
   ├── presentation_symptoms
   │   ├── fig_frequency_chart_data___0.csv
   │   ├── fig_frequency_chart_metadata.json
   │   ├── fig_table_data___0.csv
   │   ├── fig_table_metadata.json
   │   ├── fig_text_data___0.csv
   │   ├── fig_text_metadata.json
   │   ├── fig_upset_data___0.csv
   │   ├── fig_upset_data___1.csv
   │   └── fig_upset_metadata.json
   ├── treatments_interventions
   │   ├── fig_frequency_chart_inter_data___0.csv
   │   ├── fig_frequency_chart_inter_metadata.json
   │   ├── fig_frequency_chart_treat_data___0.csv
   │   ├── fig_frequency_chart_treat_metadata.json
   │   ├── fig_table_data___0.csv
   │   ├── fig_table_metadata.json
   │   ├── fig_text_data___0.csv
   │   ├── fig_text_metadata.json
   │   ├── fig_upset_inter_data___0.csv
   │   ├── fig_upset_inter_data___1.csv
   │   ├── fig_upset_inter_metadata.json
   │   ├── fig_upset_treat_data___0.csv
   │   ├── fig_upset_treat_data___1.csv
   │   └── fig_upset_treat_metadata.json
   └── visuals
       ├── enrolment_details
       │   ├── fig_sunburst.png
       │   └── fig_text.png
       ├── lesion_assessment
       │   ├── fig_table_data___0.csv
       │   └── fig_text.png
       ├── outcomes_complications
       │   ├── fig_frequency_chart.png
       │   ├── fig_table_data___0.csv
       │   ├── fig_text.png
       │   └── fig_upset.png
       ├── presentation_demogcomor
       │   ├── fig_dual_stack_pyramid.png
       │   ├── fig_frequency_chart.png
       │   ├── fig_table_data___0.csv
       │   ├── fig_text.png
       │   └── fig_upset.png
       ├── presentation_symptoms
       │   ├── fig_frequency_chart.png
       │   ├── fig_table_data___0.csv
       │   ├── fig_text.png
       │   └── fig_upset.png
       └── treatments_interventions
           ├── fig_frequency_chart.png
           ├── fig_table_data___0.csv
           ├── fig_text.png
           └── fig_upset.png

   14 directories, 75 files
