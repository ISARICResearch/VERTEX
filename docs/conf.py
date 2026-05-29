# -- IMPORTS --

# -- Standard libraries --
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

from datetime import datetime

# -- 3rd party libraries --
# -- Internal libraries --
import vertex
from vertex import __version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

author = "ISARIC"
copyright = f"ISARIC, {datetime.now().year}"
description = """
              A web-based application designed to operationalize ISARIC ARC and tailor ISARIC CRFs to disease outbreaks.
              """
github_url = "https://github.com"
github_repo = f"{github_url}/ISARICResearch/VERTEX"
github_version = "main"
# pypi_project = ''
project = vertex.__name__.upper()
release = f"v{__version__}"
public_app_url = f"https://{project}.isaric.org"

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

# Global substitutions available in every source page - not all used
rst_epilog = f"""
.. |author|                 replace:: **{author}**
.. |copyright|              replace:: **{copyright}**
.. |project|                replace:: **{project}**
.. |project_description|    replace:: {description}
.. |release|                replace:: **{release}**
.. |vrelease|               replace:: **{release}**
.. |github_release_target|  replace:: https://github.com/ISARICResearch/{project.upper()}/releases/tag/{release}
.. |public_app_url|         replace:: {public_app_url}
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
    "sphinxcontrib.email",
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
    "sphinxext.remoteliteralinclude",
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
    "member-order": "alphabetical",
    "private-members": False,
    "special-members": "",
}

# Sphinx autodoc autosummary settings
autosummary_generate = False

# Numpydoc settings
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False
numpydoc_xref_param_type = False

# Intersphinx mappings to reference external documentation domains - only
# the Python
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "Python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    #'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

# Obfuscate all mailto links in the docs sources, instead using the `email`
# role from `sphinxcontrib-email`.
email_automode = True

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
# pygments_style = "sphinx"

# A list of prefixes that are ignored when creating the module index.
# (new in Sphinx 0.6)
modindex_common_prefix = ["vertex."]

# Not currently required but will be useful later once all public
# library docstrings are complete, with doctest examples
doctest_global_setup = "import vertex"

# If this is True, the ``todo`` and ``todolist`` extension directives
# produce output, else they produce nothing. The default is ``False``.
todo_include_todos = True

# -- Project file data variables ---------------------------------------------

html_title = f"{project.upper()} {release}"

# HTML global context for templates
html_context = {
    "authors": author,
    "copyright": copyright,
    "default_mode": "dark",
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "ISARICResearch",
    "github_repo": f"{project.upper()}",
    "github_version": "main",
    "doc_path": "docs",
    "conf_path": "docs/conf.py",
    "project": project,
    "project_description": description,
    "release": release,
    "release_target": f"https://github.com/ISARICResearch/{project.upper()}/releases/tag/{release}",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# General (non-theme) HTML output options
# Custom deployment domain required here
# html_baseurl = ''

# HTML theme options
html_theme = "furo"
html_theme_options = {
    # Customisation to make the site light-only (disable dark mode).
    "dark_css_variables": {
        # Taken from: https://github.com/pradyunsg/furo/blob/c682d5d3502f3fa713c909eebbf9f3afa0f469d9/src/furo/assets/styles/variables/_colors.scss
        "color-problematic": "#b30000",
        # Base Colors
        "color-foreground-primary": "black",  # for main text and headings
        "color-foreground-secondary": "#5a5c63",  # for secondary text
        "color-foreground-muted": "#646776",  # for muted text
        "color-foreground-border": "#878787",  # for content borders
        "color-background-primary": "white",  # for content
        "color-background-secondary": "#f8f9fb",  # for navigation + ToC
        "color-background-hover": "#efeff4ff",  # for navigation-item hover
        "color-background-hover--transparent": "#efeff400",
        "color-background-border": "#eeebee",  # for UI borders
        "color-background-item": "#ccc",  # for "background" items (eg: copybutton)
        # Announcements
        "color-announcement-background": "#000000dd",
        "color-announcement-text": "#eeebee",
        # Brand colors
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2a5adf",
        # Highlighted text (search)
        "color-highlighted-background": "#ddeeff",
        # GUI Labels
        "color-guilabel-background": "#ddeeff80",
        "color-guilabel-border": "#bedaf580",
        # API documentation
        "color-api-keyword": "var(--color-foreground-secondary)",
        "color-highlight-on-target": "#ffffcc",
        # Admonitions
        "color-admonition-background": "transparent",
        # Cards
        "color-card-border": "var(--color-background-secondary)",
        "color-card-background": "transparent",
        "color-card-marginals-background": "var(--color-background-hover)",
        # Code blocks
        "color-code-foreground": "black",
        "color-code-background": "#f8f9fb",
    },
    "footer_icons": [
        {
            "name": "BRIDGE@GitHub",
            "url": "https://github.com/ISARICResearch/VERTEX",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,  # noqa : E501
            "class": "",
        },
    ],
}

# Override the default sidebar listing by commenting out the ethical ads sidebar.
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        # "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# Force pygments style in dark mode back to the light variant
pygments_dark_style = "tango"

html_logo = "_static/isaric-logo.png"

# Relative path (from the ``docs`` folder) to the static files folder - so
# ``_static`` should be one level below ``docs``.
html_static_path = ["_static"]

# Custom CSS file(s)
html_css_files = [
    "css/custom.css",
]

# Timestamp format for the last page updated time
html_last_updated_fmt = "%b %d, %Y"

# Show link to ReST source on HTML pages
html_show_sourcelink = True

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = True

# Output file base name for HTML help builder - use the project name
htmlhelp_basename = "vertex"
