.. _getting-started:

Getting Started
---------------

The latest release is |vrelease| - download the source code from `here <https://github.com/ISARICResearch/VERTEX/releases/tag/v2.0.0>`_, or clone the repository. There is also a `public app <https://vertex.isaric.org>`_ that is freely available to use. Or you can build and run your own local version (in a standalone Docker container), as described :doc:`here <app>`.


.. _requirements:

Requirements
~~~~~~~~~~~~

In terms of Python any version from ``3.11`` or higher should be fine. Specific dependencies are listed in the `project TOML <https://github.com/ISARICResearch/VERTEX/blob/main/pyproject.toml>`_. These dependencies include an ISARIC dependency named `ISARICAnalytics <https://github.com/ISARICResearch/ISARICAnalytics/>`_ that is used for REDCap data extraction, descriptive analytics, and all visualisations. See the `ISARICAnalytics documentation <https://isaricanalytics.readthedocs.io>`_ for more details.
