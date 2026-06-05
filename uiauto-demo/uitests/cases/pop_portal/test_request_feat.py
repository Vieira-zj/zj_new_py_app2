import logging

import pytest
from playwright.sync_api import expect

from tools import constant, new_browser_context, scroll_to_element
from uitests.cases.pop_portal import conftest
from uitests.pages import RequestSearchPage, RequestViewPage
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
    @pytest.mark.skip(reason="only for testing")
    def test_ui_hello_world(self):
        page = self.context.new_page()
        feature_task = FeatureRequestTask(page)
        feature_task.open_request_home_page()

    @pytest.mark.ui
    def test_view_request(self):
        page = self.context.new_page()
        view_page = RequestViewPage(page)
        feature_task = FeatureRequestTask(page)

        request_id = "SPCPMTEST-103671"
        feature_task.open_request_view_page(request_id)

        # verify page title
        page_title = view_page.get_title(
            request_name="test jira fields", request_id=request_id
        )
        expect(page_title).to_be_visible()

        # verify request background
        bg_textarea = view_page.get_request_background_textarea_raw()
        expect(bg_textarea).to_be_visible()
        logger.info("request background: %s", bg_textarea.text_content())

        scroll_to_element(bg_textarea)
        page.wait_for_timeout(timeout=constant.wait_short)
        page.screenshot(path="/tmp/test/request_view.png")

    @pytest.mark.ui
    def test_search_request(self):
        page = self.context.new_page()
        search_page = RequestSearchPage(page)
        feature_task = FeatureRequestTask(page)

        request_id = "SPCPMTEST-103671"
        feature_task.open_request_home_page()
        feature_task.search_request_by_id(request_id)
        page.wait_for_load_state()

        # verify search requests
        table_row = search_page.get_search_table_row(request_id="SPCPMTEST-103671")
        expect(table_row).to_be_visible()

        got_request_id = table_row.locator("td").nth(0).inner_text()
        assert request_id == got_request_id, "search request id is not matched"

        page.wait_for_timeout(timeout=constant.wait_short)
        page.screenshot(path="/tmp/test/request_search.png", full_page=True)


if __name__ == "__main__":
    pass
