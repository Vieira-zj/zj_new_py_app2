import datetime
from typing import Protocol


class UILocatorAssertions(Protocol):

    def to_be_visible(
        self,
        *,
        visible: bool | None = None,
        timeout: float | datetime.timedelta | None = None,
    ) -> None: ...


def test_ui_asserts_for_pw():
    from playwright.sync_api import LocatorAssertions

    def verify(la: UILocatorAssertions):
        la.to_be_visible()
        if isinstance(la, LocatorAssertions):
            la.to_be_checked()

    verify(LocatorAssertions(""))


if __name__ == "__main__":
    pass
