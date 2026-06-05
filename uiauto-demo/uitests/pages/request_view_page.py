from playwright.sync_api import Locator, Page

from tools.ui_decorator import pw_web
from uitests.pages import fixture


class RequestViewPage:
    def __init__(self, page: Page) -> None:
        self.page = page

    def get_url(self) -> str:
        return f"{fixture.base_request_url}/view-request"

    @pw_web(locator='get_by_text("{request_name} - {request_id}")')
    def get_title(self, request_name: str, request_id: str) -> Locator:
        _, _ = request_name, request_id
        return Locator("")

    def get_request_background_textarea_web(self) -> Locator:
        return self.page.locator('div[data-test-id="background"]').locator("textarea")
