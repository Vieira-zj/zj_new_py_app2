from playwright.sync_api import Locator, Page

from tools import constant, pw_web
from uitests.pages import fixture


class RequestViewPage:
    def __init__(self, page: Page) -> None:
        self.page = page

    def get_url(self) -> str:
        return f"{fixture.base_request_url}/view-request"

    def open(self, request_id: str):
        self.page.goto(
            f"{self.get_url()}?issue_key={request_id}", timeout=constant.wait_long
        )

    @pw_web(locator='get_by_text("{request_name} - {request_id}")')
    def get_title(self, request_name: str, request_id: str) -> Locator:
        _, _ = request_name, request_id
        return Locator("mockup")

    @pw_web(locator="locator('div[data-test-id=\"background\"]')")
    def get_request_background_div(self) -> Locator:
        return Locator("mockup")

    def get_request_background_textarea_raw(self) -> Locator:
        return self.get_request_background_div().locator("textarea")
