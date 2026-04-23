import os
import traceback
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from playwright.sync_api import Playwright, sync_playwright
from pydantic import SecretStr

load_dotenv()


def image_view_by_llm(image_url, prompt) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=SecretStr(api_key),
    )

    completion = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"prompt: {prompt}, image_url={image_url}"),
        ],
    )
    return completion.model_dump()


def auto_main(pw: Playwright):
    code_image_id = "#demoCaptcha_CaptchaImage"
    secret_code_input_id = "#captchaCode"
    validate_button_id = "#validateCaptchaButton"
    validation_result_text_id = "#validationResult"

    browser = pw.chromium.launch(headless=False)
    try:
        page = browser.new_page()
        page.goto("https://captcha.com/demos/features/captcha-demo.aspx")

        page.wait_for_selector(code_image_id)
        img_src = page.locator(code_image_id).get_attribute("src")

        prompt = "请识别图像中的验证码并仅返回结果字符, 里面的内容只会由数字和字母组成, 千万不要输出任何标点符号内容. 并且可能会有一些干扰线, 请仔细甄别后给出结果."
        secret_code = image_view_by_llm(img_src, prompt)["code"]
        print("secret code in image:", secret_code)

        page.type(secret_code_input_id, secret_code, delay=100)
        page.click(validate_button_id)
        page.wait_for_selector(validation_result_text_id)
        result = page.inner_text(validation_result_text_id)
        print("validation result:", result)
    except Exception as e:
        print("got error:", e)
        print(traceback.format_exc())
    finally:
        browser.close()


if __name__ == "__main__":
    with sync_playwright() as p:
        auto_main(p)
