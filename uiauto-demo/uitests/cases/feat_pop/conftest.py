import logging

import pytest
from playwright.sync_api import Playwright, sync_playwright

logger = logging.getLogger(__name__)

PW_APP: Playwright


@pytest.fixture(scope="session")
def uiauto_session_setup():
    logger.info("before pop ui auto session")
    global PW_APP
    PW_APP = sync_playwright().start()

    yield

    logger.info("after pop ui auto session")
    PW_APP.stop()
