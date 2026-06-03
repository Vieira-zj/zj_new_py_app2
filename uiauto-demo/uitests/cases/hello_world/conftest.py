import logging
from typing import Final

import pytest

logger = logging.getLogger(__name__)


# Env


class HttpConnection:
    def __init__(self) -> None:
        self.host: str = "pytest.jin.example"
        self.port: int = 8001
        self._timeout: int = 0

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        self._timeout = value

    def close(self):
        logger.info("close connection")


ECHO_TEXT: Final[str] = "hello pytest"

HTTP_CONN: HttpConnection | None = None


# Fixture


def pytest_configure(config):
    config.run_flag = "jin.test"


# fixture scopes: session, module, class, function
@pytest.fixture(scope="session", autouse=True)
def session_setup(pytestconfig):
    logger.info("before session")
    logger.info("global config: run_flag=%s", pytestconfig.run_flag)

    yield

    logger.info("after session")


@pytest.fixture(scope="session")
def db_setup():
    db = {
        "url": "mysql.test",
        "port": 3306,
    }
    yield db


@pytest.fixture(scope="session")
def http_conn_setup():
    global HTTP_CONN
    HTTP_CONN = HttpConnection()
    yield
    HTTP_CONN.close()


@pytest.fixture(scope="class")
def http_conn_timeout():
    global HTTP_CONN  # pylint: disable=W0602:global-variable-not-assigned
    logger.info("before conn timeout fixutre")
    assert HTTP_CONN, "http connection is not init"
    HTTP_CONN.timeout = 100

    yield

    logger.info("after conn timeout fixutre")


@pytest.fixture
def user_info():
    return {"id": 1, "name": "Foo"}
