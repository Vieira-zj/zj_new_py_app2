import os
import time
from urllib.parse import urljoin

import requests
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


def test_download_py(pw: Playwright):
    base_url = "https://www.python.org"
    download_path = os.path.join(os.path.expanduser("~"), "Download")

    browser = pw.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(base_url, timeout=60_000)

    # 点击导航栏 Downloads
    page.click("a:has-text('Downloads')")
    # 等待页面加载
    page.wait_for_load_state("networkidle")

    # 找到 Download Python xxx 按钮 (模糊匹配文字)
    download_btn = page.query_selector("a.button:has-text('Download Python')")
    if download_btn is None:
        print("no download button found")
        return

    # 下载 Python 安装文件
    href = download_btn.get_attribute("href")
    full_download_url = urljoin(base_url, href)
    print("get python download url:", full_download_url)

    file_name = full_download_url.split("/")[-1] + ".pkg"
    save_path = os.path.join(download_path, file_name)

    file_data = requests.get(full_download_url, timeout=3).content
    with open(save_path, "wb") as f:
        f.write(file_data)
    print("saved python pkg:", save_path)

    browser.close()


if __name__ == "__main__":
    with sync_playwright() as p:
        # test_webauto_01(p)
        test_webauto_02(p)

        # test_download_py(p)

    print("web auto finished")
