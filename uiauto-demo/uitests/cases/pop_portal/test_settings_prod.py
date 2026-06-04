import logging

import pytest
from playwright.sync_api import expect

from tools import new_browser_context, scroll_to_element
from uitests.cases.pop_portal import conftest

logger = logging.getLogger(__name__)


class TestProductSettings:
    @classmethod
    def setup_class(cls):
        logger.info("before class")
        cls.context = new_browser_context(conftest.PW_APP)
        cls.context.add_cookies(conftest.auth_cookies)

    @classmethod
    def teardown_class(cls):
        logger.info("after class")
        cls.context.close()

    @pytest.mark.ui
    def test_goto_product_settings(self):
        page = self.context.new_page()
        page.goto(
            f"{conftest.base_request_url}/requests-list", timeout=conftest.wait_long
        )

        # open Settings menu
        # settings_menu = page.get_by_text("Settings", exact=True)
        settings_menu = page.get_by_role("menuitem", name="setting Settings")
        expect(settings_menu).to_be_visible()
        scroll_to_element(settings_menu)
        settings_menu.click()

        # open POP menu
        pop_menu = page.get_by_role("menuitem", name="POP", exact=True)
        expect(pop_menu).to_be_visible()
        pop_menu.click()

        # open Porduct Line page
        product_menu = page.get_by_role("menuitem", name="Product Line")
        expect(pop_menu).to_be_visible()
        product_menu.click()
        page.wait_for_load_state()

        # verify page title
        product_title = page.get_by_text("Product Line Management")
        expect(product_title).to_be_visible()

        page.wait_for_timeout(conftest.wait_short)
        page.screenshot(path="/tmp/test/product_setting.png")

    @pytest.mark.ui
    def test_search_products(self):
        page = self.context.new_page()
        page.goto(
            f"{conftest.base_settings_url}/product-line", timeout=conftest.wait_long
        )

        product_title = page.get_by_text("Product Line Management")
        expect(product_title).to_be_visible()

        # input for search
        bizline_input = page.locator("id=business_line_id_list")
        expect(bizline_input).to_be_visible()

        bizline_input.click()
        page.keyboard.type("POP")
        page.keyboard.press("Enter")
        product_title.click()

        # search
        search_button = page.locator('button:has-text("Search")')
        expect(search_button).to_be_visible()
        search_button.click()
        page.wait_for_load_state()

        # verify search products
        products = page.locator('td[data-test-id="name"]')
        expect(products).to_have_count(6)
        for item in products.all():
            logger.info("product line: %s", item.text_content())

        page.wait_for_timeout(conftest.wait_short)
        page.screenshot(path="/tmp/test/search_product.png")


if __name__ == "__main__":
    pass
