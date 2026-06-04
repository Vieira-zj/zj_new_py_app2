import logging
import os

import pytest
from playwright.sync_api import Playwright, sync_playwright

logger = logging.getLogger(__name__)

# Env

auth_cookies: list = [
    {
        "name": "space_auth_live",
        "value": os.getenv("SPACE_AUTH"),
        "domain": os.getenv("POP_MAIN_DOMAIN"),
        "path": "/",
    },
    {
        "name": "SPC_SEC_SI",
        "value": os.getenv("SPC_SEC_SI"),
        "domain": os.getenv("POP_TEST_DOMAIN"),
        "path": "/",
    },
    {
        "name": "SSO_C",
        "value": os.getenv("SSO_C"),
        "domain": os.getenv("POP_TEST_DOMAIN"),
        "path": "/",
    },
]

PW_APP: Playwright


# Pytest Fixutre


@pytest.fixture(scope="session", autouse=True)
def uiauto_session_setup():
    logger.info("before pop ui auto session")
    global PW_APP
    PW_APP = sync_playwright().start()

    yield

    logger.info("after pop ui auto session")
    PW_APP.stop()
