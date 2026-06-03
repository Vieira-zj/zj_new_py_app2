import os
from typing import Final
from venv import logger

from playwright.sync_api import BrowserContext, Playwright

# Env

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


# Helper


def new_browser_context(pw: Playwright) -> BrowserContext:
    # logger.info("chromium.executable_path: %s", pw.chromium.executable_path)
    browser = pw.chromium.launch(
        headless=False,
        args=["--start-maximized"],
        executable_path=os.getenv("CHROMIUM_PATH"),
    )
    context = browser.new_context()
    context.add_cookies(auth_cookies)
    return context
