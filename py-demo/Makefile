ENV := test

default: help

# cli: make help ENV=live
help:
	@uv --version
	@echo "\nEnv: $(ENV)"

init-env:
	uv venv --python 3.12.1
	uv sync --frozen

run-main:
	uv run main.py

run-eg:
	uv run examples/py_base.py

dep-tree:
	uv tree

dump-requirements:
	uv export --format requirements.txt > requirements.txt
