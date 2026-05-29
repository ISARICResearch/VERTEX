.. _getting-started:

Getting Started
---------------

The latest release, which is |vrelease|, containing the source code, can be downloaded from `GitHub <https://github.com/ISARICResearch/VERTEX/releases/tag/v2.0.0>`_, or the repository can be cloned with Git or an Git-integrated IDE of your choice, e.g. VS Code. There is no public Python package associated with the VERTEX repository.

VERTEX is essentially an app, and there is a `public version <https://vertex.isaric.org>`_ that is freely available to use. Or you can build and run your own local version in a standalone Docker container, as described :doc:`here <app>`.


.. _requirements:

Requirements
~~~~~~~~~~~~

The main VERTEX requirements are Python ``3.11+`` and the
specific dependencies listed in the ``[project]`` section of the `project TOML <https://github.com/ISARICResearch/VERTEX/blob/main/pyproject.toml>`_.

If you're running VERTEX locally :doc:`in Docker <app>` then these dependencies (and their sub-dependencies) will be pre-installed inside the container, so no direct user installation is required.

If you're running VERTEX directly on your system as a Python (Plotly Dash) app then you need to ensure all of these dependencies are installed in your environment with the current version pins, **before** running the app. If you haven't already done this, then you can do this either by installing the project in editable mode, via :command:`pip install -e .`, which will also install the main ``vertex`` package in your environment, or a direct installation using a package manager such as `Astral UV <https://docs.astral.sh/uv/>`_. For example, if using UV you may consider using the :command:`uv sync` command as described `here <https://docs.astral.sh/uv/concepts/projects/sync/#syncing-the-environment>`_.

The VERTEX dependencies listed in the TOML include an ISARIC dependency named `ISARICAnalytics <https://github.com/ISARICResearch/ISARICAnalytics/>`_ that is used for REDCap data extraction, descriptive analytics, and all visualisations. See the `ISARICAnalytics documentation <https://isaricanalytics.readthedocs.io>`_ for more details.
