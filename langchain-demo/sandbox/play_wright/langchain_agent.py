import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from sandbox_manager import get_global_manager

load_dotenv()

# ============ LangChain Tools ============


@tool
def create_browser_sandbox(
    template_name: Optional[str] = None, idle_timeout: int = 3000
) -> str:
    try:
        manager = get_global_manager()
        info = manager.create(template_name=template_name, idle_timeout=idle_timeout)
        result = f"""✅ Sandbox create successful.
📋 Sandbox info:
- ID: {info['sandbox_id']}
- CDP URL: {info['cdp_url']}
"""
        vnc_url = info.get("vnc_url")
        if vnc_url:
            result += f"- VNC URL: {vnc_url}\n\n"
        return result
    except Exception as e:
        return f"create Sandbox instance failed: {str(e)}"


@tool
def get_sandbox_info() -> str:
    try:
        manager = get_global_manager()
        info = manager.get_info()
        result = f"""✅ Sandbox create successful.
📋 Sandbox info:
- ID: {info['sandbox_id']}
- CDP URL: {info['cdp_url']}
"""
        vnc_url = info.get("vnc_url")
        if vnc_url:
            result += f"- VNC URL: {vnc_url}\n\n"
        return result
    except RuntimeError as e:
        return f"{str(e)}"
    except Exception as e:
        return f"get Sandbox info failed: {str(e)}"


class NavigateInput(BaseModel):
    url: str = Field(
        description="browser URL, must be start with 'http://' or 'https://'"
    )
    wait_until: str = Field(
        default="load",
        description="wait page status: load, domcontentloaded, networkidle",
    )
    timeout: int = Field(
        default=30000, description="timeout by milli secs, default 30000"
    )


@tool(args_schema=NavigateInput)
def navigate_to_url(
    url: str,
    wait_until: Optional[Literal["load", "domcontentloaded", "networkidle"]] = "load",
    timeout: int = 30000,
) -> str:
    try:
        manager = get_global_manager()
        if not manager.is_active():
            return "no active sandbox, pls create"
        if not url.startswith(("http://", "https://")):
            return f"invalid input url: {url}"
        cdp_url = manager.get_cdp_url()
        if not cdp_url:
            return "cannot get CDP URL"

        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(cdp_url)
                pages = browser.contexts[0].pages if browser.contexts else []
                if pages:
                    page = pages[0]
                else:
                    page = browser.new_page()
                page.goto(url, wait_until=wait_until, timeout=timeout)
                title = page.title()
                return f"successful navigate to: {url}\n📄 page title: {title}\n"
        except ImportError:
            return "pls install playwright (pip install playwright)"
        except Exception as e:
            return f"navigate to url failed: {str(e)}"
    except Exception as e:
        return f"perform operation failed: {str(e)}"


@tool("browser_screenshot", description="screenshot current page in browser sandbox")
def take_screenshot(filename: str = "screenshot.png") -> str:
    try:
        manager = get_global_manager()
        if not manager.is_active():
            return "no active sandbox, pls create"
        cdp_url = manager.get_cdp_url()
        if not cdp_url:
            return "cannot get CDP URL"
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp(cdp_url)
                pages = browser.contexts[0].pages if browser.contexts else []
                if pages:
                    page = pages[0]
                else:
                    return "no opened page"
                page.screenshot(path=filename)
                return f"screenshot save successful: {filename}"
        except ImportError:
            return "pls install playwright (pip install playwright)"
        except Exception as e:
            return f"take screenshot failed: {str(e)}"
    except Exception as e:
        return f"perform operation failed: {str(e)}"


@tool
def extract_table_data(url: str) -> str:
    manager = get_global_manager()
    cdp_url = manager.get_info()["cdp_url"]
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(cdp_url)
        page = browser.contexts[0].pages[0]
        page.goto(url)
        tables = page.query_selector_all("table")
        return f"found {len(tables)} table"


@tool(
    "destroy_sandbox",
    description="destory current sandbox instance, and release resource",
)
def destroy_sandbox() -> str:
    try:
        manager = get_global_manager()
        return manager.destroy()
    except Exception as e:
        return f"destory sandbox failed: {str(e)}"


# ============ LangChain Agent ============


def get_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("pls add env DASHSCOPE_API_KEY")
    return api_key


def create_browser_agent(system_prompt: Optional[str] = None):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("pls add env DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name = os.getenv("QWEN_MODEL", "qwen-plus")

    model = ChatOpenAI(
        model=model_name, api_key=get_api_key, base_url=base_url, temperature=0.7
    )

    tools = [
        create_browser_sandbox,
        get_sandbox_info,
        navigate_to_url,
        take_screenshot,
        destroy_sandbox,
    ]

    if system_prompt is None:
        system_prompt = """你是一个浏览器自动化助手, 可以使用 sandbox 来访问和操作网页.

当用户需要访问网页时, 请按以下步骤操作:
1. 首先创建或获取 sandbox (如果还没有)
2. 使用 navigate_to_url 导航到目标网页
3. 执行用户请求的操作
4. 如果需要, 可以截取截图

重要提示:
- 创建 sandbox 后, 会返回 VNC URL, 用户可以使用它实时查看浏览器操作
- 所有操作都会在 VNC 中实时显示, 方便调试和监控
- sandbox 可以在多轮对话中复用, 不要在一轮对话完成后就销毁
- 只有在用户明确要求销毁时才使用 destroy_sandbox 工具
- 不要主动建议用户销毁 sandbox, 除非用户明确要求
- 请始终用中文回复, 确保操作准确, 高效."""

    agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)
    return agent
