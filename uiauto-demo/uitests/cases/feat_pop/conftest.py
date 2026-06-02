import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    logger.info("before pop ui auto session")

    yield

    logger.info("after pop ui auto session")
