import logging

import pytest

from tools import new_browser_context
from uitests.cases.pop_portal import conftest
from uitests.tasks import FeatureRequestTask

logger = logging.getLogger(__name__)


class TestHelloWorldPage:
    @classmethod
    def setup_class(cls):
        logger.info("before hello class")
        cls.context = new_browser_context(conftest.PW_APP)
        cls.context.add_cookies(conftest.auth_cookies)

    @classmethod
    def teardown_class(cls):
        logger.info("after hello class")
        cls.context.close()

    @pytest.mark.ui
    @pytest.mark.skip(reason="only for testing")
    def test_ui_hello_world(self):
        page = self.context.new_page()
        feature_task = FeatureRequestTask(page)
        feature_task.open_request_home_page()


if __name__ == "__main__":
    pass
