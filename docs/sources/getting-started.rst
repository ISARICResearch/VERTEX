
===========================
Getting Started with VERTEX
===========================

.. container::

   .. rubric:: Contents
      :name: contents

   - :ref:`Purpose <getting-started.purpose>`
   - :ref:`Introduction <getting-started.introduction>`
   - :ref:`Installation and Usage <getting-started.installation-and-usage>`
   - :ref:`VERTEX Dashboard <getting-started.dashboard>`
   - :ref:`Next Steps <getting-started.next-steps>`

.. _getting-started.purpose:

Purpose
--------
This guide provides a brief introduction to the Visual Evidence & Research Tool for Exploration (VERTEX) dashboard and the Reusable Analytical Pipelines for Infectious Diseases (RAPID) code framework. It provides instructions for how to run the base VERTEX software on your local machine.

.. _getting-started.introduction:

Introduction
------------

VERTEX is a web-based Python application that presents graphs and tables relating to research questions that need to be quickly answered during an outbreak. This helps identification of key epidemiological factors and supports data-driven decision-making.

|VERTEX|

VERTEX is open-source, which means that you can access and download the codebase from GitHub\ :sup:`1`, and create new analysis projects for your analysis. At the moment, the VERTEX dashboard can display two types of projects.

**Analysis projects** retrieve data from a REDCap database using the API, preprocess this patient-level data\ :sup:`2` and execute analysis code to answer research questions from a statistical analysis plan. The output of this analysis code is a series of figures and tables, which are presented in the dashboard, grouped together in **insight panels**.

For structuring analysis code, we borrow a concept called Reproducible Analytical Pipelines\ :sup:`4`. This is intended to make analysis reproducible, efficient and auditable. We extend this concept to **Reusable Analytical Pipelines for Infectious Diseases (RAPIDs)**\ :sup:`4`, which supports the transportation of these pipelines to answer the similar research questions in similar contexts.

**Static projects** do not execute the analysis pipelines, but reproduce the figures and tables directly from aggregated (not patient-level) data files\ :sup:`5`. These files can be provided directly or can be automatically created when the users runs an analysis project.

:sup:`1`\ https://github.com/ISARICResearch/VERTEX/blob/main/README.md
:sup:`2`\ We make some assumptions about the structure of the REDCap project and a core subset of the variables within the data, this is described further in the follow-up training guides listed at the end.
:sup:`3`\ https://analysisfunction.civilservice.gov.uk/support/reproducible-analytical-pipelines/
:sup:`4`\ We are currently developing a standalone package to support the development of RAPIDs. This is intended to make development of pipelines more user-friendly and to include more detailed documentation and testing. Users will initially be able to customise or develop analysis code outside of the VERTEX dashboard, but can still integrate the outputs within the dashboard.
:sup:`5`\ Each type of figure requires aggregated data in a specific structure, with accompanying metadata to provide other input parameters. We are currently refining this, so we do not have up-to-date documentation about each figure type. However, we can support this if you have aggregated data and you want to present results within a dashboard.

.. _getting-started.installation-and-usage:

Installation and Usage
-----------------------
To use VERTEX locally, clone the VERTEX GitHub repository\ :sup:`6` and make sure that you have Python installed. There are several ways to run the VERTEX code, and depending on how your preferences for coding, you may also find it useful to have the following installed: git\ :sup:`7`, Docker\ :sup:`8` and VSCode\ :sup:`9`.

We will demonstrate how to run the code at the command line using Docker and in VSCode without Docker, though we recommend using Docker if possible\ :sup:`10`.

.. rubric:: Docker at the command line
   :name: docker

On the command line, navigate to the repository. Depending on how you cloned the repository, this may be in your Downloads folder or in the current working directory.

We recommend using Docker if you want to run VERTEX at the command line. You can build and run a Docker container for VERTEX with the following:

::

   docker build -t isaric-vertex .

   docker run -it --rm -v "$PWD":/app -p 8050:8050 isaric-vertex gunicorn --workers 1 --reload --bind 0.0.0.0:8050 vertex.descriptive_dashboard:server

.. rubric:: VSCode without Docker
   :name: docker

Open the VERTEX folder within VSCode. You need to create an environment\ :sup:`11` from the VERTEX requirements. The simplest way to do this is Ctrl + Shift + p (or Command + Shift + p on MacOS), then click "Python: Create environment...", ".venv" and choose a Python version that is >3.9. This will install a virtual environment
using the requirements listed in ``pyproject.toml``.

Next, you need to add a file ``.vscode/launch.json`` with the following:

::


   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "ISARIC VERTEX",
         "type": "python",
         "request": "launch",
         "module": "vertex.descriptive_dashboard",
         "justMyCode": false,
         "env": {
           "PYTHONPATH": "${workspaceFolder}"
         }
       }
     ]
   }


You should now be able to run the application in debugging mode ("Run → Start Debugging") or without debugging ("Run → Run Without Debugging").

:sup:`6`\ https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
:sup:`7`\ https://git-scm.com/install/
:sup:`8`\ https://docs.docker.com/engine/install/
:sup:`9`\ https://code.visualstudio.com/docs/languages/python
:sup:`10`\ See https://code.visualstudio.com/docs/containers/overview for using Docker within VSCode.
:sup:`11`\ https://code.visualstudio.com/docs/python/environments

.. _getting-started.dashboard:

VERTEX Dashboard
----------------
Once the application is running at the terminal or in VSCode, you will be able to open the local VERTEX dashboard at
http://127.0.0.1:8050. You will see a menu on the left, and a world map with a colorbar.

|VERTEX dashboard|

At the top of the menu is listed the current open project. You can switch between active projects using the dropdown immediately below. The current active projects are **analysis projects** that use low-fidelity synthetic datasets and **static projects** showing the results from completed ISARIC analyses.

The map shows the number of participants that were enrolled in this project by country. You can hover over each country for a more precise count of the number of participants.

Below the project selection box in the menu are a set of tabs, which are mostly specific to the selected project. **Analysis projects**, which retrieve the up-to-date patient-level data from a REDCap database and directly execute analysis code, have a tab called **Filters and Controls**. If you click on this, it will open up to show a set of core variables: sex at birth, age, country, admission date and outcome. You can select different combinations, which will filter the patient-level data and update the map accordingly.

For all projects, the remaining tabs should each be relevant to a research question or a moment/event in the patient's journey. Each tab can be expanded to show one or more buttons, which may break this down further into group of related variables e.g. demographics and comorbidities. Each button will then open an object called an
**insight panel**.

Insight panels show the outputs of an analysis pipeline. At this time, this will mostly involve descriptive tables and summary figures displaying counts or proportions. We will continue to add new analyses as we build a larger collection of RAPIDs. As with the base page, **analysis projects** will have a tab for **Filters and Controls** and the code will be executed again if you filter the participants. This is not possible for **static projects**, which
display fixed outputs.

|Insight panels|

.. _getting-started.next-steps:

Next steps
----------
This guide has introduced VERTEX and how to run the code locally.

VERTEX is intended to run multiple projects. If you have REDCap database that was created using ISARIC BRIDGE and have API access for your database, then you should be able add a new project to your local VERTEX and display outputs from your analysis in a similar manner to the existing projects.

For more information about creating a new project in VERTEX, please refer to the following guides: `Creating a new VERTEX project <https://isaricresearch.github.io/Training/vertex_new_project.html>`__. We will add more detailed technical guides about creating new insight panels and contributing to RAPIDs in due course.

.. container:: footer

   Licensed under a `Creative Commons Attribution-ShareAlike 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`__ International License by `ISARIC <https://isaric.org/>`__ on behalf of Oxford University.

.. |VERTEX| image:: https://github.com/ISARICResearch/Training/raw/main/docs/assets/vertex_starting-insight-panel.png
.. |VERTEX dashboard| image:: https://github.com/ISARICResearch/Training/raw/main/docs/assets/vertex_starting-dashboard.png
.. |Insight panels| image:: https://github.com/ISARICResearch/Training/raw/main/docs/assets/vertex_starting-insight-panel.png
