from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def langchain_help():
    print("langchain / langgraph base examples.")


def test_fake_chat_model_01():
    model = GenericFakeChatModel(messages=iter(["hello", "world"]))
    print(model.invoke("any").content)
    print(model.invoke("any").content)


def test_fake_chat_model_02():
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
    print(fake_model.invoke("any"))


if __name__ == "__main__":
    # test_agent_with_fake_chat_model_01()
    test_fake_chat_model_02()
