# Python Package Demo

## Create Project

```sh
uv init --lib py-pkg-demo
```

## Verify Package

1. init py env

```sh
mkdir -p /tmp/test/py-pkg-demo-test
cd /tmp/test/py-pkg-demo-test

uv venv
uv pip install /tmp/test/dist/py_pkg_demo-0.1.0-py3-none-any.whl
```

2. exec

```sh
# invoke lib
uv run python -c "from py_pkg_demo import pkg_help; pkg_help()"

# run exec
uv run mypy --help

uv run mypy ls --verbose /tmp/test
uv run mypy size /tmp/test/output.json
uv run mypy find --key=main ~/Downloads/tmps
```

