from playwright.sync_api import Locator, Page

from tools.ui_decorator import pw_web


class LeftNavMenu:

    def __init__(self, page: Page) -> None:
        self.page = page

    @pw_web(locator='get_by_role("menuitem", name="setting Settings")')
    def get_settings_menuitem(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator='get_by_role("menuitem", name="POP", exact=True)')
    def get_settings_pop_menuitem(self) -> Locator:
        return Locator("mockup")

    @pw_web(locator='get_by_role("menuitem", name="Product Line")')
    def get_settings_pop_product_menuitem(self) -> Locator:
        return Locator("mockup")
