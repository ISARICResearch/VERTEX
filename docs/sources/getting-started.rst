.. _getting-started:

Getting Started
---------------

The latest release is |vrelease| - download the source code from `here <https://github.com/ISARICResearch/VERTEX/releases/tag/v2.0.0>`_, or clone the repository. There is also a `public app <https://vertex.isaric.org>`_ that is freely available to use. Or you can build and run your own local version in a standalone Docker container, as described :doc:`here <app>`.


.. _requirements:

Requirements
~~~~~~~~~~~~

The main VERTEX requirements if you're trying to run the app locally outside Docker are Python ``3.11+`` and the
specific dependencies listed in the ``[project]`` section of the `project TOML <https://github.com/ISARICResearch/VERTEX/blob/main/pyproject.toml>`_. You can either install these dependencies by installing the project in editable mode, via :command:`pip install -e .`, which will also install the main ``vertex`` package in your environment, or manually install the dependencies using a package manager such as `Astral UV <https://docs.astral.sh/uv/>`_. For example, if using UV you may consider using the :command:`uv sync` command as described `here <https://docs.astral.sh/uv/concepts/projects/sync/#syncing-the-environment>`_.

The VERTEX dependencies listed in the TOML include an ISARIC dependency named `ISARICAnalytics <https://github.com/ISARICResearch/ISARICAnalytics/>`_ that is used for REDCap data extraction, descriptive analytics, and all visualisations. See the `ISARICAnalytics documentation <https://isaricanalytics.readthedocs.io>`_ for more details.
