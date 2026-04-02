# -- IMPORTS --

# -- Standard libraries --
import os
import sys
from datetime import datetime

# -- 3rd party libraries --

# -- Internal libraries --


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

import vertex
from vertex import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

author = "ISARIC"
copyright = f"ISARIC, {datetime.now().year}"
description = """
              Dash-based VERTEX dashboard for ISARIC.
              """
github_url = "https://github.com"
github_repo = f"{github_url}/ISARICResearch/VERTEX"
github_version = "main"
# pypi_project = ''
project = vertex.__name__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Define master TOC
master_doc = "index"

# Native docs language
language = "en"

# Minimum required version of Sphinx - not required
# needs_sphinx >= '7.2.5'

# Set primary domain to null
primary_domain = None

# Global substitutions
rst_epilog = f"""
.. |author|                 replace:: **{author}**
.. |copyright|              replace:: **{copyright}**
.. |docs_url|               replace:: ''
.. |project|                replace:: **{project}**
.. |project_description|    replace:: {description}
.. |release|                replace:: **{release}**
.. |github_release_target|  replace:: https://github.com/ISARICResearch/VERTEX/releases/tag/{release}
"""

# Publish author(s)
show_authors = True

# Sphinx extensions: not all of these are used or required, but they are still
# listed here if requirements change.
extensions = [
    "jupyter_sphinx",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "nb2plots",
    "numpydoc",
    "sphinx.ext.autodoc",
    #'sphinx.ext.autosectionlabel',
    #'sphinx.ext.autosummary',
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.extlinks",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    #'sphinx.ext.linkcode',
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

# Autodoc settings -
#     For more on all available autodoc defaults see
#         https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_default_options
autodoc_default_options = {
    "exclude-members": "",
    "member-order": "bysource",
    "private-members": False,
    "special-members": "__init__,__new__",
}

# Sphinx autodoc autosummary settings
autosummary_generate = False

# Numpydoc settings
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False
numpydoc_xref_param_type = False

# Intersphinx mappings to reference external documentation domains - no
# current references, but these have been listed in case relevant new
# new references are added.
intersphinx_mapping = {
    #'dash': ('https://dash.plotly.com/', None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "Python": ("https://docs.python.org/3", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    #'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    "statsmodels": ("https://www.statsmodels.org/stable", None),
}

# Static template paths
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of prefixes that are ignored when creating the module index. (new in Sphinx 0.6)
modindex_common_prefix = ["vertex."]

doctest_global_setup = "import vertex"

# If this is True, the ``todo`` and ``todolist`` extension directives
# produce output, else they produce nothing. The default is ``False``.
todo_include_todos = True

# -- Project file data variables ---------------------------------------------

# HTML global context for templates
html_context = {
    "authors": author,
    "copyright": copyright,
    "default_mode": "dark",
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "ISARICResearch",
    "github_repo": "VERTEX",
    "github_version": "main",
    "doc_path": "docs",
    "conf_path": "docs/conf.py",
    "project": project,
    "project_description": description,
    "release": release,
    "release_target": f"https://github.com/ISARICResearch/VERTEX/releases/tag/{release}",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# General (non-theme) HTML output options
# Custom deployment domain required here
# html_baseurl = ''

# HTML theme options
html_theme = "furo"
html_theme_options = {}

# html_logo = '_static/logo.png'

# Relative path (from the ``docs`` folder) to the static files folder - so
# ``_static`` should be one level below ``docs``.
html_static_path = ["_static"]

# Custom CSS file(s) - currently source the Font Awesome CSS classes to support
# Font Awesome icons. for more information see:
#
#     https://sphinx-design.readthedocs.io/en/latest/badges_buttons.html#fontawesome-icons
#
# html_css_files = []

# Timestamp format for the last page updated time
html_last_updated_fmt = "%b %d, %Y"

# Show link to ReST source on HTML pages
html_show_sourcelink = True

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = True

# Output file base name for HTML help builder - use the project name
htmlhelp_basename = "vertex"
