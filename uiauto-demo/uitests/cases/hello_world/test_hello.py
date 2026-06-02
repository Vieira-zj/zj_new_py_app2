import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class", autouse=True)
def hello_world_class_setup():
    logger.info("before fixture class")

    yield

    logger.info("after fixture class")


class TestHelloWorld:
    def setup_method(self, method):
        logger.info("setup_method")
        test_name = "%s::%s" % (self.__class__.__name__, method.__name__)
        logger.info("run test case: %s", test_name)

    def teardown_method(self):
        logger.info("teardown_method")

    @pytest.mark.smoke
    def test_hello_case01(self):
        logger.info("case: hello world")

    @pytest.mark.smoke
    def test_hello_case02(self, pytestconfig):
        logger.info("pytest config: run_flag=%s", pytestconfig.run_flag)

    def test_hello_case03(self):
        logger.info("non-smoke case")
