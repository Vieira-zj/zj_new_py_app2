import os
from typing import Final

from playwright.sync_api import BrowserContext, Playwright, sync_playwright

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


def create_browser_context() -> tuple[Playwright, BrowserContext]:
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=False,
        args=["--start-maximized"],
    )
    context = browser.new_context()
    context.add_cookies(auth_cookies)
    return pw, context
