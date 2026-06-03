import logging
import time

import pytest

from tools import new_browser_context, scroll_to_element
from uitests.cases.feat_pop import conftest

logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("uiauto_session_setup")
class TestFeatureRequest:
    @classmethod
    def setup_class(cls):
        logger.info("before class")
        cls.base_url = conftest.base_url_of_request
        cls.context = new_browser_context(conftest.PW_APP)
        cls.context.add_cookies(conftest.auth_cookies)

    @classmethod
    def teardown_class(cls):
        logger.info("after class")
        time.sleep(3)
        cls.context.close()

    @pytest.mark.ui
    def test_view_request(self):
        request_id: str = "SPCPMTEST-103671"

        page = self.context.new_page()
        page.goto(
            f"{self.base_url}/view-request?issue_key={request_id}",
            timeout=conftest.wait_page_timeout,
        )

        # verify page title
        title = page.locator("div.CwTEbLVARGQ-")
        title_text = title.text_content()
        assert title_text, "page title is not found"
        assert request_id in title_text, "incorrect page title text"
        logger.info("page title: %s", title.text_content())

        # verify request background
        bg_div = page.locator('div[data-test-id="background"]')
        bg_textarea = bg_div.locator("textarea")
        assert bg_textarea, "request background is not found"
        logger.info("request background: %s", bg_textarea.text_content())

        scroll_to_element(bg_textarea)
        time.sleep(1)
        page.screenshot(path="/tmp/test/request_view.png")

    @pytest.mark.ui
    def test_search_request(self):
        request_id = "SPCPMTEST-103671"

        page = self.context.new_page()
        page.goto(f"{self.base_url}/requests-list", timeout=conftest.wait_page_timeout)

        # verify page title
        title = page.wait_for_selector(
            'span[data-test-id="title"]', timeout=conftest.wait_ui_element_timeout
        )
        assert title, "page title is not found"
        logger.info("page title: %s", title.text_content())

        # input and select request id
        request_id_input = page.locator('input[id="rc_select_0"]')
        request_id_input.click()
        page.keyboard.type(request_id)

        with page.expect_response(
            lambda response: "/request/parent/search" in response.url
        ) as resp:
            logger.info("request search resp: %s", resp.value)
            page.keyboard.press("Enter")

        # click search
        # search_icon = page.get_by_role("button", name="search")
        search_icon = page.locator('button:has-text("Search")')
        assert search_icon, "search button is not found"
        logger.info("search button: %s", search_icon.text_content())
        search_icon.click()

        # verify search results
        page.wait_for_load_state()
        table_row = page.locator('tr[data-row-key="SPCPMTEST-103671"]')
        assert table_row, "search table row is not found"

        got_request_id = table_row.locator("td").nth(0).inner_text()
        assert request_id == got_request_id, "search request id is not matched"

        page.screenshot(path="/tmp/test/request_search.png", full_page=True)

    @pytest.mark.ui
    def test_nav_to_product_config(self):
        page = self.context.new_page()
        page.goto(f"{self.base_url}/requests-list", timeout=conftest.wait_page_timeout)

        # click Settings menu
        main_menu = page.locator(
            'ul[id="rc-menu-uuid-66498-1-Request Management-popup]"'
        )
        settings_menu_item = main_menu.locator("li").nth(8)
        assert settings_menu_item, "settings menu is not found"
        settings_menu_item.click()

        # click POP menu
        settings_menu = page.locator('ul[id="rc-menu-uuid-66498-1-Settings-popup"]')
        pop_menu_item = settings_menu.locator('span:has-text("POP")')
        assert settings_menu_item, "pop menu is not found"
        pop_menu_item.click()


if __name__ == "__main__":
    # test_search_feat_request()
    pass
