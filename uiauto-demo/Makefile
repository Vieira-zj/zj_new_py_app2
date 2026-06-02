default: help

help:
	@uv --version

run-main:
	uv run --env-file .env main.py

dep-tree:
	uv tree

dump-requirements:
	uv export --format requirements.txt > requirements.txt

run-tests:
	uv run pytest -c pytest.ini uitests/cases/hello_world
# 	uv run pytest uitests/cases/hello_world/test_hello.py::TestHelloWorld
# 	uv run pytest uitests/cases/test_pop_feat_request.py::TestFeatureRequest

run-tests-with-mark:
	uv run pytest -m smoke uitests/cases/hello_world

collect-tests:
	uv run pytest --collect-only
