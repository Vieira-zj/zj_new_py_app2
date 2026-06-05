default: help

help:
	@uv --version

run-main:
	uv run --env-file .env main.py

dep-tree:
	uv tree

dump-requirements:
	uv export --format requirements.txt > requirements.txt

collect-tests:
	uv run pytest --collect-only

collect-markers:
	uv run pytest --markers

collect-fixtures:
	uv run pytest --fixtures

collect-fixture-per-test:
	uv run pytest --fixtures-per-test

run-tests:
# 	uv run pytest -s -c pytest.ini uitests/cases/hello_world
	uv run pytest uitests/cases/hello_world/test_hello.py::TestHelloWorld::test_hello_case03

run-ui-tests:
# 	uv run pytest uitests/cases/pop_portal
	uv run pytest uitests/cases/pop_portal/test_request_feat.py::TestFeatureRequest::test_ui_hello_world

run-tests-with-mark:
	uv run pytest -s -m smoke uitests/cases/hello_world
# 	uv run pytest -s -m "not smoke" uitests/cases/hello_world
# 	uv run pytest -s -m "smoke and api" uitests/cases/hello_world
