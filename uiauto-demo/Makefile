default: help

help:
	@uv --version

run-main:
	uv run --env-file .env main.py

dep-tree:
	uv tree

dump-requirements:
	uv export --format requirements.txt > requirements.txt

run-test:
# 	uv run pytest -v -o log_cli=true uitests/cases/test_hello_world.py::TestHelloWorld::test_hello
	uv run pytest -o log_cli=true uitests/cases/test_pop_feat_request.py::TestFeatureRequest

collect-tests:
	uv run pytest --collect-only
