import os
from typing import Final

import pytest
from dotenv import load_dotenv
from playwright.sync_api import BrowserContext, Playwright, sync_playwright

from utils import init_logger

load_dotenv()

# Env Variables

wait_page_timeout: Final[int] = 10_000
wait_ui_element_timeout: Final[int] = 3_000

base_url_of_feat_request: str = (
    f"https://{os.getenv("POP_TEST_DOMAIN")}/request-management"
)

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


# Pytest Fixture

logger = init_logger(__name__)


def pytest_configure(config):
    config.run_flag = "jin.test"


@pytest.fixture(scope="module", autouse=True)
def module_setup(pytestconfig):
    logger.info("before module: run_flag=%s", pytestconfig.run_flag)

    yield

    logger.info("after module")


# Helper


def create_browser_context() -> tuple[Playwright, BrowserContext]:
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=False,
        args=["--start-maximized"],
    )
    context = browser.new_context()
    context.add_cookies(auth_cookies)
    return pw, context
