ENV := test

default: help

# cli: make help ENV=live
help:
	@uv --version
	@echo "ENV: $(ENV)"

init-env:
	uv venv --python 3.12.1
	uv sync --frozen

run-main:
	uv run main.py

run-tests:
# 	uv run pytest -v tests/
	uv run pytest -v tests/test_os_utils.py::test_run_shell_cmd

dep-tree:
	uv tree

dump-requirements:
	uv export --format requirements.txt > requirements.txt
