import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class", autouse=True)
def hello_class_setup(pytestconfig):
    logger.info("before class")
    logger.info("global config: run_flag=%s", pytestconfig.run_flag)
    yield
    logger.info("after class")


@pytest.fixture(scope="function")
def mock_hello_fixture():
    logger.info("before custom hello fixture")
    yield
    logger.info("after custom hello fixture")


@pytest.mark.usefixtures("db_setup")
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

    @pytest.mark.usefixtures("mock_hello_fixture", "user_info")
    def test_hello_case02(self, user_info: dict):
        logger.info("case: use custom fixture")
        logger.info("user info: %s", user_info)

    def test_hello_case03(self, pytestconfig, db_setup: dict):
        logger.info("pytest config: run_flag=%s", pytestconfig.run_flag)
        logger.info("db: %s", db_setup)
