SHELL := /bin/bash

REPO := https://github.com/ISARICResearch/VERTEX

PACKAGE_NAME := vertex
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
HEAD := $(shell git rev-parse --short=8 HEAD)
PACKAGE_VERSION := $(shell grep __version__ vertex/__init__.py | cut -d '=' -f 2 | xargs)

PROJECT_ROOT := $(PWD)

TESTS_ROOT := $(PROJECT_ROOT)/tests

DOCS_ROOT := $(PROJECT_ROOT)/docs
DOCS_BUILD := $(DOCS_ROOT)/_build
DOCS_BUILD_HTML := $(DOCS_ROOT)/_build/html


# --- Git ---
#
# Git staging
git-stage:
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Staging new, modified, deleted and/or renamed files in Git"
	git status -uno | grep modified | tr -s ' ' | cut -d ' ' -f 2 | xargs git add && \
	git status -uno | grep deleted | tr -s ' ' | cut -d ' ' -f 2 | xargs git add -A && \
	git status -uno

# --- Housekeeping ---
.PHONY: clean
clean:
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Deleting all temporary files"
	rm -fr dist/* docs/_build/* .pytest_cache *.pyc *__pycache__* ./dist/* ./build/* *.egg-info*
	python3 -m pip uninstall -y isaric-vertex

# --- Version commands ---
#
# A simple file-based version check for the installed package (local, sdist or wheel)
version-check:
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Checking installed package version (if it is installed)"
	python3 -c "import os; os.chdir('src/arc'); from __init__ import __version__; print(__version__); os.chdir('../')"

# Just display the version
version-extract:
	echo "$(PACKAGE_VERSION)"

# --- Member inspection of Python files ---
#
# Parameterised command to lexicographically list all callables (functions,
# classes) in path-specified Python modules including:
#
# functions:
#
#     list-members MEMBER_REGEX="def" FILE_PATH="/path/to/py/file"
#
# classes:
#
#     list-members MEMBER_REGEX="class" FILE_PATH="/path/to/py/file"
#
# functions or classes (inclusive):
#
#     list-members MEMBER_REGEX="def\|class" FILE_PATH="/path/to/py/file"
#
# where the '|' represents an OR operator for grep that needs to be escaped with the backslash '\'.
list-callables:
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Lexicographical isting of all callable members ($(MEMBER_REGEX)) of a Python file"
	grep "$(MEMBER_REGEX)" $(FILE_PATH) | sort | cut -d ' ' -f 2 | cut -d '(' -f 1

# --- Dependency management ---
sync-deps-exact:
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Syncing all package + development dependencies with lockfile, removing unrelated dependencies"
	rm -f uv.lock && \
	uv sync --verbose --all-groups --no-editable --no-install-project --no-cache --refresh --no-managed-python

sync-deps-inexact:
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Syncing all package + development dependencies with lockfile, preserving unrelated dependencies"
	rm -f uv.lock && \
	uv sync --verbose --all-groups --no-editable --no-install-project --no-cache --refresh --inexact

# --- Package artifacts ---
#
# Only source distributions are required, no wheel.
.PHONY: sdist
sdist: clean
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Building a source distribution of VERTEX"
	uv build --verbose --sdist && ls -al dist/*.tar.gz

.PHONE: uninstall
uninstall: clean
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Uninstalling any existing installed VERTEX distribution"

# --- Documentation ---
.PHONY: clean
docs: clean
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Building Sphinx docs (using the Sphinx Makefile in ./docs/)"
	make -C docs html

# --- Pre-commit ---
.PHONY: clean
pre-commit: clean
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Running pre-commit hooks"
	pre-commit run --all-files

# --- Tests ---
#
# Unit tests - use the `MARKER` variable to indicate markers, e.g.
# "critical or high", or "medium or low". Note that in the `test` target
# command below, the `MARKER` variable must be quoted to prevent expansion
# in case of spaces in the marker.
.PHONY: test
test: clean
	@echo "$(PACKAGE_NAME)[$(BRANCH)@$(HEAD)]: Running unit tests + measuring coverage"
	PYTHONPATH=src uv run --verbose --active -m pytest \
	                               -q -m "$(MARKER)" \
	                               --cache-clear \
	                               --capture=no \
	                               --code-highlight=yes \
	                               --color=yes \
	                               --cov=src \
	                               --cov-report=term-missing:skip-covered \
	                               -ra \
	                               --tb=native \
	                               --verbosity=3 \
	                               $(TESTS_PATH)
