import time

from playwright.sync_api import Playwright, sync_playwright

# playwright chromium env:
# uv run playwright install chromium


def test_webauto_01(pw: Playwright):
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://www.baidu.com")

    page.fill("#chat-textarea", value="playwright")
    page.press("#chat-textarea", key="Enter")

    page.wait_for_load_state("networkidle")
    print("result:", page.title())

    page.screenshot(path="/tmp/test/search.png", full_page=True)

    # time.sleep(2)
    browser.close()


def test_webauto_02(pw: Playwright):
    """inject js into web page by playwright."""
    browser = pw.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.baidu.com")

    # exec js and get return value
    title = page.evaluate("document.title")
    print(f"page title: {title}")

    # exec js with arguments
    result = page.evaluate("([a, b]) => a + b", [2, 4])
    print(f"sum: {result}")

    # update dom
    page.evaluate(
        """
const div = document.createElement('div');
div.id = 'injected';
div.textContent = 'Injected by Playwright!';
div.style.cssText = 'background: yellow; padding: 20px; font-size: 24px;';
document.body.prepend(div);
"""
    )

    # add init script (runs before page loads)
    # page.add_init_script("""window.myCustomVar = 'Hello from init script';""")

    # add external script
    # page.add_script_tag(url="https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js")

    page.screenshot(path="/tmp/test/search.png")

    time.sleep(2)
    browser.close()


if __name__ == "__main__":
    with sync_playwright() as p:
        # test_webauto_01(p)
        test_webauto_02(p)

    print("web auto finished")
