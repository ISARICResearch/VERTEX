.. _getting-started:

Getting Started
===============

The latest release, which is |vrelease|, containing the source code, can be downloaded from `GitHub <https://github.com/ISARICResearch/VERTEX/releases/tag/v2.0.0>`_, or the repository can be cloned with Git or an Git-integrated IDE of your choice, e.g. VS Code. There is no public Python package associated with the VERTEX repository.

VERTEX is essentially an app, and there is a `public version <https://vertex.isaric.org>`_ that is freely available to use. Or you can build and run your own local version in a standalone Docker container, as described :doc:`here <app>`.


.. _requirements:

Requirements
------------

The requirements for using VERTEX as a local app depend on how you want to run it:

* If you want to run VERTEX in a **Docker container**, as described :ref:`here <docker>`, then the only requirement is `Docker Desktop <https://www.docker.com/products/docker-desktop/>`_. The Docker build process will ensure, via the `Dockerfile <https://github.com/ISARICResearch/VERTEX/blob/main/Dockerfile>`_, that all the app dependencies (and their sub-dependencies) are pre-installed inside the image used to run the container. **Docker is the recommended way of running VERTEX locally** as it results in more stable, reproducible builds.

* If you want to run VERTEX directly on your system / host as a Python **Plotly Dash** app then you need to ensure all of the `project dependencies <https://github.com/ISARICResearch/VERTEX/blob/main/pyproject.toml#L62>`_ are installed in your VERTEX environment, as described :ref:`below <plotly-dash-set-up>`, **before** running the app. Please ensure that your environment is using a minimum of Python ``3.12+`` (although ``3.11`` should also be OK). For further information on the requirement dependencies see the `project TOML <https://github.com/ISARICResearch/VERTEX/blob/main/pyproject.toml>`_.

.. _plotly-dash-set-up:

Setting up the VERTEX environment to run as a Plotly Dash app
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. In a Python 3.12+ virtual environment install `Astral uv <https://docs.astral.sh/uv/>`_ with :program:`pip`:

.. code:: shell

   python3 -m pip install uv

By default UV will make all dependency-related changes inside a new :file:`.venv` subfolder within the working directory - if this is OK then proceed to the next step. **If not** then either export the path of the preferred (e.g. pre-existing or working) virtual environment with the ``UV_PROJECT_ENVIRONMENT`` environment variable:

.. code:: shell

   export UV_PROJECT_ENVIRONMENT="/path/to/preferred/virtual/env"

or use the ``--active`` flag on the relevant UV command, which will usually be :command:`uv sync` - see the `documentation <https://docs.astral.sh/uv/concepts/projects/sync/>`_ for more details.

2. Install (or sync) all the project dependencies into the working environment with the command:

.. code:: shell

   uv sync --verbose --active --all-groups --no-install-project --no-cache --refresh --inexact

.. note::

   With the  ``--inexact`` flag :command:`uv sync` preserves dependencies installed in the target environment that are unrelated to the dependency structure implied by the project TOML, while with the ``--exact`` flag its execution will result in a set of dependencies (and sub-dependencies) in the working environment that exactly match the dependency structure implied by the project TOML, including the removal of unrelated dependencies. To be on the safe side, it is advisable to run :command:`uv sync` with ``--inexact``, unless explicitly required for some reason.
