from playwright.sync_api import Page, expect

from tools import constant, scroll_to_element
from uitests.pages import LeftNavMenu, ProductSettingsPage


class ProductSettingsTask:

    def __init__(self, page: Page) -> None:
        self.page = page
        self.prod_settings_page = ProductSettingsPage(page)
        self.left_nav_menu = LeftNavMenu(page)

    def open_product_settings_page(self):
        self.page.goto(self.prod_settings_page.get_url(), timeout=constant.wait_long)

        product_title = self.prod_settings_page.get_page_title()
        expect(product_title).to_be_visible()

    def nav_to_pop_product_settings(self):
        # open Settings menu
        # settings_menu = page.get_by_text("Settings", exact=True)
        settings_menu = self.left_nav_menu.get_settings_menuitem()
        expect(settings_menu).to_be_visible()
        scroll_to_element(settings_menu)
        settings_menu.click()

        # open POP menu
        pop_menu = self.left_nav_menu.get_settings_pop_menuitem()
        expect(pop_menu).to_be_visible()
        pop_menu.click()

        # open Porduct Line page
        product_menu = self.left_nav_menu.get_settings_pop_product_menuitem()
        expect(pop_menu).to_be_visible()
        product_menu.click()
        self.page.wait_for_load_state()

    def search_by_bizline_id(self, bizline_id: str):
        product_title = self.page.get_by_text("Product Line Management")
        expect(product_title).to_be_visible()

        # input for search
        bizline_input = self.prod_settings_page.get_bizline_input()
        expect(bizline_input).to_be_visible()

        bizline_input.click()
        self.page.keyboard.type(bizline_id)
        self.page.keyboard.press("Enter")
        product_title.click()

        # click search
        search_button = self.prod_settings_page.get_search_button()
        expect(search_button).to_be_visible()
        search_button.click()
        self.page.wait_for_load_state()


if __name__ == "__main__":
    pass
