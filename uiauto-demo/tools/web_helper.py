import os

from playwright.sync_api import BrowserContext, Locator, Page, Playwright


def new_browser_context(pw: Playwright) -> BrowserContext:
    # logger.info("chromium.executable_path: %s", pw.chromium.executable_path)
    browser = pw.chromium.launch(
        headless=False,
        args=["--start-maximized"],
        executable_path=os.getenv("CHROMIUM_PATH"),
    )
    context = browser.new_context()
    return context


def scroll_to_element(ele: Locator):
    ele.evaluate("""
el => el.scrollIntoView({
    behavior: 'smooth',
    block: 'center'
})
""")


def scroll_to_page_bottom(page: Page):
    for _ in range(3):
        old_height = page.evaluate("document.body.scrollHeight")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(1000)
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == old_height:
            return


if __name__ == "__main__":
    pass
