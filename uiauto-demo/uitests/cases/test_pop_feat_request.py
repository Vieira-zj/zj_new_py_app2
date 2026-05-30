import time

import conftest

from utils import init_logger

logger = init_logger(__name__)


class TestFeatureRequest:
    @classmethod
    def setup_class(cls):
        logger.info("before class: init browser context")
        cls.base_url = conftest.base_url_of_feat_request
        cls.pw, cls.context = conftest.create_browser_context()

    @classmethod
    def teardown_class(cls):
        logger.info("after class: wait and close browser")
        time.sleep(5)
        cls.context.close()
        cls.pw.stop()

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
        assert title_text, "page title is empty"
        assert request_id in title_text, "incorrect page title text"
        logger.info("page title: %s", title.text_content())

        # TODOs: verify detail

    def test_search_request(self):
        request_id = "SPCPMTEST-103671"

        # open page
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


if __name__ == "__main__":
    # test_search_feat_request()
    pass
