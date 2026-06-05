from playwright.sync_api import Locator, Page

from tools import constant, pw_web
from uitests.pages import fixture


class RequestSearchPage:

    def __init__(self, page: Page) -> None:
        self.page = page

    def get_url(self) -> str:
        return f"{fixture.base_request_url}/requests-list"

    def open(self):
        self.page.goto(self.get_url(), timeout=constant.wait_long)

    @pw_web(locator='get_by_text("All Requests")')
    def get_title(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator="locator('input[id=\"rc_select_0\"]')")
    def get_request_id_input(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator="locator('button:has-text(\"Search\")')")
    def get_search_button(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator="locator('tr[data-row-key=\"{request_id}\"]')")
    def get_search_table_row(self, request_id: str) -> Locator:
        _ = request_id
        return Locator("mockup")


if __name__ == "__main__":
    pass
