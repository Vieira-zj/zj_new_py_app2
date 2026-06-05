import datetime
from typing import Any, Literal, Pattern, Protocol, Self

from playwright.sync_api import Locator, expect

from uimodel import UILocatorAssertions

type LocatorRoles = Literal[
    "alert",
    "alertdialog",
    "application",
    "article",
    "banner",
    "blockquote",
    "button",
    "caption",
    "cell",
    "checkbox",
    "code",
    "columnheader",
    "combobox",
    "complementary",
    "contentinfo",
    "definition",
    "deletion",
    "dialog",
    "directory",
    "document",
    "emphasis",
    "feed",
    "figure",
    "form",
    "generic",
    "grid",
    "gridcell",
    "group",
    "heading",
    "img",
    "insertion",
    "link",
    "list",
    "listbox",
    "listitem",
    "log",
    "main",
    "marquee",
    "math",
    "menu",
    "menubar",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "meter",
    "navigation",
    "none",
    "note",
    "option",
    "paragraph",
    "presentation",
    "progressbar",
    "radio",
    "radiogroup",
    "region",
    "row",
    "rowgroup",
    "rowheader",
    "scrollbar",
    "search",
    "searchbox",
    "separator",
    "slider",
    "spinbutton",
    "status",
    "strong",
    "subscript",
    "superscript",
    "switch",
    "tab",
    "table",
    "tablist",
    "tabpanel",
    "term",
    "textbox",
    "time",
    "timer",
    "toolbar",
    "tooltip",
    "tree",
    "treegrid",
    "treeitem",
]


class UILocator(Protocol):

    @property
    def first(self) -> Self: ...
    @property
    def last(self) -> Self: ...

    def nth(self, index: int) -> Self: ...
    def all(self) -> list[Self]: ...
    def count(self) -> int: ...

    def click(self) -> None: ...
    def fill(self, value: str) -> None: ...
    def clear(self) -> None: ...

    def get_by_role(
        self,
        role: LocatorRoles,
        *,
        name: str | Pattern[str] | None = None,
        exact: bool | None = None,
    ) -> Self: ...

    def get_by_test_id(self, test_id: str | Pattern[str]) -> Self: ...
    def get_by_text(
        self,
        text: str | Pattern[str],
        *,
        exact: bool | None = None,
    ) -> Self: ...
    def locator(self, selector_or_locator: str | Self) -> Self: ...

    def inner_text(self) -> str: ...
    def text_content(
        self,
        *,
        timeout: float | datetime.timedelta | None = None,
    ) -> str | None: ...

    def evaluate(self, expression: str) -> Any: ...


def ui_expect(locator: UILocator) -> UILocatorAssertions:
    if isinstance(locator, Locator):
        return expect(locator)
    raise ValueError("not support locator type")


def test_ui_locator_for_pw():
    def verify(locator: UILocator):
        ui_expect(locator).to_be_visible()

    verify(locator=Locator(""))


if __name__ == "__main__":
    pass
