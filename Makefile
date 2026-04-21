PYTHON_VERSION := $(shell cat .python-version)
UV_PATH    := export PATH=$$HOME/.local/bin:$$PATH
# PyGIMLi lives in a conda env (no macOS ARM wheel on PyPI).
# `make setup` expects the damforge conda env to already exist with pygimli installed.
# One-time bootstrap (already done): brew install --cask miniforge &&
#   conda create -n damforge python=3.12 && conda install -n damforge -c gimli -c conda-forge pygimli
CONDA_PYTHON := /opt/homebrew/Caskroom/miniforge/base/envs/damforge/bin/python

.PHONY: install-uv
install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: install-python
install-python:
	$(UV_PATH) && uv python install $(PYTHON_VERSION)

.PHONY: install-deps
install-deps:
	$(UV_PATH) && uv venv --system-site-packages --python $(CONDA_PYTHON) && uv sync --all-extras --all-groups

.PHONY: install-pre-commit
install-pre-commit:
	$(UV_PATH) && uv run pre-commit install

.PHONY: setup
setup: install-uv install-python install-deps install-pre-commit
	echo "PYTHONPATH=$${PWD}" >> .env

.PHONY: test
test:
	uv run pytest tests

.PHONY: coverage
coverage:
	uv run coverage run -m pytest
	uv run coverage report --fail-under=80

.PHONY: coverage-report
coverage-report:
	uv run coverage run -m pytest
	uv run coverage html

.PHONY: docs
docs:
	uv run mkdocs serve

.PHONY: build
build:
	uv build



