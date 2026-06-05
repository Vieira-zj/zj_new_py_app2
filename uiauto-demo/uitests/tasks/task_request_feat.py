import logging

from playwright.sync_api import Page, expect

from tools import constant
from uitests.pages import RequestSearchPage, RequestViewPage

logger = logging.getLogger(__name__)


class FeatureRequestTask:

    def __init__(self, page: Page) -> None:
        self.page = page
        self.request_search_page = RequestSearchPage(page)
        self.request_view_page = RequestViewPage(page)

    def open_request_view_page(self, request_id: str):
        self.page.goto(
            f"{self.request_view_page.get_url()}?issue_key={request_id}",
            timeout=constant.wait_long,
        )

    def open_request_home_page(self):
        self.page.goto(self.request_search_page.get_url(), timeout=constant.wait_long)
        title = self.request_search_page.get_title()
        expect(title).to_be_visible()

    def search_request_by_id(self, request_id: str):
        request_id_input = self.request_search_page.get_request_id_input()
        expect(request_id_input).to_be_visible()
        request_id_input.click()

        # input and select request id
        self.page.keyboard.type(request_id)
        with self.page.expect_response(
            lambda response: "/request/parent/search" in response.url
        ) as resp:
            logger.info("request search resp: %s", resp.value)
            self.page.keyboard.press("Enter")

        # click search
        search_button = self.request_search_page.get_search_button()
        expect(search_button).to_be_visible()
        search_button.click()


if __name__ == "__main__":
    pass
