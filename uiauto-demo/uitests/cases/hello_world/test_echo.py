import logging

import pytest

from uitests.cases.hello_world import conftest

logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("http_conn_setup", "http_conn_timeout")
class TestEcho:
    @classmethod
    def setup_class(cls):
        logger.info("setup class")
        cls.echo_text = conftest.ECHO_TEXT
        cls.conn = conftest.HTTP_CONN

    @classmethod
    def teardown_class(cls):
        logger.info("teardown class")

    @pytest.mark.smoke
    def test_echo_text(self):
        logger.info("echo: %s", self.echo_text)

    @pytest.mark.skip(reason="jin testing")
    def test_echo_mock_skip(self):
        logger.info("this case is mark as skipped")

    @pytest.mark.xfail(reason="bug not fix")
    def test_echo_mock_fail(self):
        assert False, "mock assert failed"

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "a,b,result",
        [
            (1, 2, 3),
            (2, 3, 5),
            (3, 4, 7),
        ],
    )
    def test_echo_parametrize(self, a, b, result):
        assert a + b == result

    def test_echo_use_conn_fixture(self):
        assert self.conn
        logger.info(
            "http conn fixture: host=%s, port=%d, timeout=%d",
            self.conn.host,
            self.conn.port,
            self.conn.timeout,
        )
