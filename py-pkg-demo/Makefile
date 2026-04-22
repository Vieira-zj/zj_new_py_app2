default: help

help:
	@uv --version

build:
	@rm -r dist/
	@uv build

run-tests:
	@uv run pytest -v -s tests/

run-test:
	uv run pytest -v -s tests/test_cli.py
