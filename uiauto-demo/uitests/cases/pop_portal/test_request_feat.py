import logging

import pytest
from playwright.sync_api import expect

from tools import new_browser_context, scroll_to_element
from uitests.cases.pop_portal import conftest

logger = logging.getLogger(__name__)


class TestFeatureRequest:
    @classmethod
    def setup_class(cls):
        logger.info("before class")
        cls.base_url = conftest.base_request_url
        cls.context = new_browser_context(conftest.PW_APP)
        cls.context.add_cookies(conftest.auth_cookies)

    @classmethod
    def teardown_class(cls):
        logger.info("after class")
        cls.context.close()

    @pytest.mark.ui
    def test_view_request(self):
        request_id: str = "SPCPMTEST-103671"

        page = self.context.new_page()
        page.goto(
            f"{self.base_url}/view-request?issue_key={request_id}",
            timeout=conftest.wait_long,
        )

        # verify page title
        title = page.get_by_text(f"test jira fields - {request_id}")
        expect(title).to_be_visible()

        # verify request background
        bg_textarea = page.locator('div[data-test-id="background"]').locator("textarea")
        expect(bg_textarea).to_be_visible()
        logger.info("request background: %s", bg_textarea.text_content())

        scroll_to_element(bg_textarea)
        page.wait_for_timeout(timeout=conftest.wait_short)
        page.screenshot(path="/tmp/test/request_view.png")

    @pytest.mark.ui
    def test_search_request(self):
        request_id = "SPCPMTEST-103671"

        page = self.context.new_page()
        page.goto(f"{self.base_url}/requests-list", timeout=conftest.wait_long)

        # verify page title
        title = page.get_by_text("All Requests")
        expect(title).to_be_visible()

        # input and select request id
        request_id_input = page.locator('input[id="rc_select_0"]')
        expect(request_id_input).to_be_visible()
        request_id_input.click()

        page.keyboard.type(request_id)
        with page.expect_response(
            lambda response: "/request/parent/search" in response.url
        ) as resp:
            logger.info("request search resp: %s", resp.value)
            page.keyboard.press("Enter")

        # click search
        search_button = page.locator('button:has-text("Search")')
        expect(search_button).to_be_visible()
        search_button.click()

        # verify search requests
        page.wait_for_load_state()
        table_row = page.locator('tr[data-row-key="SPCPMTEST-103671"]')
        expect(table_row).to_be_visible()

        got_request_id = table_row.locator("td").nth(0).inner_text()
        assert request_id == got_request_id, "search request id is not matched"

        page.screenshot(path="/tmp/test/request_search.png", full_page=True)


if __name__ == "__main__":
    pass
