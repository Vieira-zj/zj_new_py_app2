import logging

import pytest

from uitests.cases.hello_world import conftest

logger = logging.getLogger(__name__)


class TestEcho:
    @classmethod
    def setup_class(cls):
        logger.info("setup class")
        cls.echo_text = conftest.echo_text

    @classmethod
    def teardown_class(cls):
        logger.info("teardown class")

    @pytest.mark.smoke
    def test_echo_case01(self):
        logger.info("echo: %s", self.echo_text)

    @pytest.mark.skip
    def test_echo_case02(self):
        logger.info("this case is mark as skipped")
