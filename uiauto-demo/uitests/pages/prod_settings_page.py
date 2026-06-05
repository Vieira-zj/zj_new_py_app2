from playwright.sync_api import Locator, Page

from tools import constant, pw_web
from uitests.pages import fixture


class ProductSettingsPage:

    def __init__(self, page: Page) -> None:
        self.page = page

    def get_url(self) -> str:
        return f"{fixture.base_settings_url}/product-line"

    def open(self):
        self.page.goto(self.get_url(), timeout=constant.wait_long)

    @pw_web(locator='get_by_text("Product Line Management")')
    def get_title(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator='locator("id=business_line_id_list")')
    def get_bizline_input(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator="locator('button:has-text(\"Search\")')")
    def get_search_button(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator="locator('td[data-test-id=\"name\"]')")
    def get_all_search_products(self) -> Locator:
        return Locator("mockup")
