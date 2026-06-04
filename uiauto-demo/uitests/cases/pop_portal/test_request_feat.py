import logging

import pytest
from playwright.sync_api import expect

from tools import constant, new_browser_context, scroll_to_element
from uitests.cases.pop_portal import conftest
from uitests.tasks import FeatureRequestTask

logger = logging.getLogger(__name__)


class TestFeatureRequest:
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
    def test_view_request(self):
        request_id: str = "SPCPMTEST-103671"

        page = self.context.new_page()
        feature_task = FeatureRequestTask(page)
        feature_task.open_request_view_page(request_id)

        # verify page title
        title = page.get_by_text(f"test jira fields - {request_id}")
        expect(title).to_be_visible()

        # verify request background
        bg_textarea = page.locator('div[data-test-id="background"]').locator("textarea")
        expect(bg_textarea).to_be_visible()
        logger.info("request background: %s", bg_textarea.text_content())

        scroll_to_element(bg_textarea)
        page.wait_for_timeout(timeout=constant.wait_short)
        page.screenshot(path="/tmp/test/request_view.png")

    @pytest.mark.ui
    def test_search_request(self):
        request_id = "SPCPMTEST-103671"

        page = self.context.new_page()
        feature_task = FeatureRequestTask(page)

        feature_task.open_request_home_page()
        feature_task.search_request_by_id(request_id)

        # verify search requests
        page.wait_for_load_state()
        table_row = page.locator('tr[data-row-key="SPCPMTEST-103671"]')
        expect(table_row).to_be_visible()

        got_request_id = table_row.locator("td").nth(0).inner_text()
        assert request_id == got_request_id, "search request id is not matched"

        page.screenshot(path="/tmp/test/request_search.png", full_page=True)


if __name__ == "__main__":
    pass
