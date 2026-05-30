import conftest
import pytest

from utils import init_logger

logger = init_logger(__name__)


@pytest.fixture(scope="class", autouse=True)
def hello_world_class_setup():
    logger.info("before class")

    yield

    logger.info("after class")


class TestHelloWorld:
    @classmethod
    def setup_class(cls):
        logger.info("setup class")
        cls.base_url = conftest.base_url_of_feat_request

    @classmethod
    def teardown_class(cls):
        logger.info("teardown class")

    def test_hello(self, pytestconfig):
        logger.info("start pytest case: hello world")
        logger.info("run_flag=%s, base_url=%s", pytestconfig.run_flag, self.base_url)
