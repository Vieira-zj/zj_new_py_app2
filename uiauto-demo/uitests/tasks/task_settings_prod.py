from playwright.sync_api import Page, expect

from tools import constant, scroll_to_element
from uitests.tasks import fixture


class ProductSettingsTask:

    def __init__(self, page: Page) -> None:
        self.page = page

    def open_product_settings_page(self):
        self.page.goto(
            f"{fixture.base_settings_url}/product-line", timeout=constant.wait_long
        )

        product_title = self.page.get_by_text("Product Line Management")
        expect(product_title).to_be_visible()

    def nav_to_pop_product_settings(self):
        # open Settings menu
        # settings_menu = page.get_by_text("Settings", exact=True)
        settings_menu = self.page.get_by_role("menuitem", name="setting Settings")
        expect(settings_menu).to_be_visible()
        scroll_to_element(settings_menu)
        settings_menu.click()

        # open POP menu
        pop_menu = self.page.get_by_role("menuitem", name="POP", exact=True)
        expect(pop_menu).to_be_visible()
        pop_menu.click()

        # open Porduct Line page
        product_menu = self.page.get_by_role("menuitem", name="Product Line")
        expect(pop_menu).to_be_visible()
        product_menu.click()
        self.page.wait_for_load_state()

    def search_by_bizline_id(self, bizline_id: str):
        product_title = self.page.get_by_text("Product Line Management")
        expect(product_title).to_be_visible()

        # input for search
        bizline_input = self.page.locator("id=business_line_id_list")
        expect(bizline_input).to_be_visible()

        bizline_input.click()
        self.page.keyboard.type(bizline_id)
        self.page.keyboard.press("Enter")
        product_title.click()

        # click search
        search_button = self.page.locator('button:has-text("Search")')
        expect(search_button).to_be_visible()
        search_button.click()
        self.page.wait_for_load_state()


if __name__ == "__main__":
    pass
