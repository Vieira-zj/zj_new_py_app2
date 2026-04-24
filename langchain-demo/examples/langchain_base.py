import json

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def langchain_help():
    import langchain

    print("langchain version:", langchain.__version__)


# LangChain


def test_langchain_expr():
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("你是乐于助人的中文助手"),
            HumanMessage("{question}"),
        ]
    )
    model = init_chat_model("deepseek:deepseek-chat")
    chain = prompt | model | StrOutputParser()

    print(chain.invoke({"question": "给我 3 条高效读论文的要点"}))


@tool
def get_weather(loc: str) -> str:
    """mock get weather api."""
    result = {"loc": loc, "weather": "cloud"}
    return json.dumps(result)


@tool
def write_to_markdown(content: str) -> str:
    """mock write content to markdown file."""
    print(f"mock write to md file: content_len={len(content)}")
    return "saved success"


def test_langchain_agent():
    sys_prompt = (
        "你是一个智能助手, 可以自己根据用户的问题来调用响应的工具帮助用户解决问题"
    )
    agent = create_agent(
        model="deepseek:deepseek-chat",
        tools=[get_weather, write_to_markdown],
        system_prompt=sys_prompt,
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage("请帮我查询北京, 杭州, 上海等地天气, 并写入本地文件")
            ]
        }
    )
    print(result["messages"])


# LangChain Middleware


def test_langchain_middleware_01():
    basic_model = init_chat_model(model="deepseek:deepseek-chat")
    advanced_model = init_chat_model(model="deepseek:deepseek-reasoner")

    @wrap_model_call
    def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
        message_count = len(request.state["messages"])
        print(f"Message count: {message_count}")

        if message_count > 1:
            model = advanced_model
        else:
            model = basic_model

        request.model = model
        print(f"selected model: {model.get_name()}")

        response = handler(request)
        print(f"actual response model: {response}")
        return response

    agent = create_agent(
        model=basic_model,
        tools=[get_weather, write_to_markdown],
        middleware=[dynamic_model_selection],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "todo"}]})
    print(result["messages"])


def test_langchain_middleware_02():
    agent = create_agent(
        model="openai:gpt-4o",
        tools=[get_weather, write_to_markdown],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_to_markdown": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                    "get_weather": False,
                }
            )
        ],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "todo"}]})
    print(result["messages"])


# Fake Tests


def test_fake_chatmodel_01():
    model = GenericFakeChatModel(messages=iter(["hello", "world"]))
    print(model.invoke("any").content)
    print(model.invoke("todo").content)


def test_fake_chatmodel_02():
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "foo",
                            "args": {"bar": "baz"},
                        }
                    ],
                ),
                "final anwser",
            ]
        )
    )
    print(fake_model.invoke("any"))
    print(fake_model.invoke("todo"))


if __name__ == "__main__":
    langchain_help()

    # test_fake_chatmodel_01()
    # test_fake_chatmodel_02()
