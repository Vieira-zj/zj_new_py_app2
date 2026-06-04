import logging
import os
from typing import Final

import pytest
from playwright.sync_api import Playwright, sync_playwright

logger = logging.getLogger(__name__)

# Env

wait_short: Final[int] = 1_000
wait_mid: Final[int] = 3_000
wait_long: Final[int] = 10_000

base_request_url = f"https://{os.getenv("POP_TEST_DOMAIN")}/request-management"
base_settings_url = f"https://{os.getenv("POP_TEST_DOMAIN")}/pop-requests/settings"

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
