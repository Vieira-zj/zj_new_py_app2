import base64
import os
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionUserMessageParam,
)

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "")


class LVMChat:
    """多模态大模型聊天类"""

    def __init__(
        self,
        api_key: str = API_KEY,
        base_url: str = BASE_URL,
        model_name: Optional[str] = "gemini-3-flash-preview",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name: str = model_name  # type: ignore
        self.conversation_history: List[
            ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam
        ] = []

    def _encode_image(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_multimodal_response(
        self, text: str, image_paths: str, use_history: bool = False
    ) -> str:
        """
        简单的图文对话

        Args:
            text: 你的问题
            image_paths: 图片路径

        Returns:
            模型的回答
        """
        # 1. 加载图片并构建当前消息
        base64_image = self._encode_image(image_paths)
        current_message = ChatCompletionUserMessageParam(
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": text},
            ],
        )

        # 2. 🔥 如果启用历史，把之前的对话也带上
        if use_history:
            messages = self.conversation_history + [current_message]
            print(f"📚 使用历史上下文，共 {len(self.conversation_history)} 条")
        else:
            messages = [current_message]

        # 3. 调用 API
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages
        )

        result = response.choices[0].message.content
        if not result:
            return ""

        # 4. 🔥 更新历史记录
        if use_history:
            self.conversation_history.append(current_message)
            self.conversation_history.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=result)
            )

        return result
