import logging

import pytest
from playwright.sync_api import expect

from tools import constant, new_browser_context
from uitests.cases.pop_portal import conftest
from uitests.pages import ProductSettingsPage
from uitests.tasks import FeatureRequestTask, ProductSettingsTask

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
        prod_settings_page = ProductSettingsPage(page)
        feature_task = FeatureRequestTask(page)
        feature_task.open_request_home_page()

        product_task = ProductSettingsTask(page)
        product_task.nav_to_pop_product_settings()

        # verify page title
        product_title = prod_settings_page.get_page_title()
        expect(product_title).to_be_visible()

        page.wait_for_timeout(constant.wait_short)
        page.screenshot(path="/tmp/test/product_setting.png")

    @pytest.mark.ui
    def test_search_products(self):
        page = self.context.new_page()
        prod_settings_page = ProductSettingsPage(page)
        product_task = ProductSettingsTask(page)

        product_task.open_product_settings_page()
        product_task.search_by_bizline_id("POP")

        # verify search products
        products = prod_settings_page.get_all_search_products()
        expect(products).to_have_count(6)
        for item in products.all():
            logger.info("product line: %s", item.text_content())

        page.wait_for_timeout(constant.wait_short)
        page.screenshot(path="/tmp/test/search_product.png")


if __name__ == "__main__":
    pass
