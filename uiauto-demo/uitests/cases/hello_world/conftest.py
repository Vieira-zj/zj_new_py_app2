import logging
from typing import Final

import pytest

logger = logging.getLogger(__name__)


# Env

echo_text: Final[str] = "hello pytest"

# Fixture


def pytest_configure(config):
    config.run_flag = "jin.test"


@pytest.fixture(scope="session", autouse=True)
def session_setup(pytestconfig):
    logger.info("before session")
    logger.info("global config: run_flag=%s", pytestconfig.run_flag)

    yield

    logger.info("after session")
