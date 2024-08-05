# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

NAME := HPI_for_MO_AutoML
PACKAGE_NAME := hpi_for_mo

DIR := "${CURDIR}"
SOURCE_DIR := ${PACKAGE_NAME}
DOCDIR := docs
INDEX_HTML := "file://${DIR}/docs/build/html/index.html"
EXAMPLES_DIR := examples
.PHONY: help install-dev check format clean docs clean-doc examples help:
	@echo "Makefile ${NAME}"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* clean            to clean any doc or build files"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* docs             to generate and view the html files, checks links"
	@echo "* examples         to run and generate the examples"
	PYTHON ?= python
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
FLAKE8 ?= flake8

install-dev:
	$(PIP) install -e ".[dev]"
	check-black:
	$(BLACK) ${SOURCE_DIR} --check || :
	$(BLACK) ${EXAMPLES_DIR} --check || :
	check-isort:
	$(ISORT) ${SOURCE_DIR} --check || :
	check-pydocstyle:
	$(PYDOCSTYLE) ${SOURCE_DIR} || :

check-flake8:
	$(FLAKE8) ${SOURCE_DIR} || :
	$(FLAKE8) ${TESTS_DIR} || :

check: check-black check-isort check-flake8 check-pydocstyle

format-black:
	$(BLACK) ${SOURCE_DIR}
	$(BLACK) ${EXAMPLES_DIR}

format-isort:
	$(ISORT) ${SOURCE_DIR}
	format: format-black format-isort

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

docs:
	$(MAKE) -C ${DOCDIR} docs
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

examples:
	$(MAKE) -C ${DOCDIR} examples
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

# Clean up any builds in ./dist as well as doc, if present
clean: clean-doc
