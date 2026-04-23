from typing import NotRequired, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


def test_langgraph_seq_01():
    class TextState(TypedDict):
        text: str

    def to_uppercase(state: TextState) -> TextState:
        print(f"[to_uppercase] Initial state: {state}")
        return TextState(text=state["text"].upper())

    def to_lowercase(state: TextState) -> TextState:
        print(f"[to_lowercase] Received state: {state}")
        return TextState(text=state["text"].lower())

    def build_graph():
        builder = StateGraph(TextState)

        # add nodes
        builder.add_node("to_uppercase", to_uppercase)
        builder.add_node("to_lowercase", to_lowercase)

        # add edges
        builder.add_edge(START, "to_uppercase")
        builder.add_edge("to_uppercase", "to_lowercase")
        builder.add_edge("to_lowercase", END)

        return builder.compile()

    graph = build_graph()
    final_state = graph.invoke(TextState(text="Hello"))
    print("final state:", final_state)


def test_langgraph_seq_02():
    class AgentState(TypedDict):
        messages: list[str]
        user_intent: Optional[str]
        next_step: Optional[str]

    def intent_analysis_node(state):
        print(f"analysis user intent {state}...")
        return {"user_intent": "repair", "next_step": "dispatch"}

    def assistant_reply_node(state):
        print("💬 generate reply...")
        return {"messages": state["messages"] + ["query results"]}

    def dispatch_ticket_node(state):
        print("dispatch ticket ...")
        # mock api call
        return {"messages": state["messages"] + ["ticket created"]}

    def router(state):
        if state["user_intent"] == "repair":
            return "dispatch"
        return "assistant"

    def build_graph():
        workflow = StateGraph(AgentState)

        workflow.add_node("intent_analysis", intent_analysis_node)
        workflow.add_node("assistant_reply", assistant_reply_node)
        workflow.add_node("dispatch_ticket", dispatch_ticket_node)

        workflow.set_entry_point("intent_analysis")
        workflow.add_conditional_edges(
            "intent_analysis",
            router,
            {"dispatch": "dispatch_ticket", "assistant": "assistant_reply"},
        )

        workflow.add_edge("assistant_reply", END)
        workflow.add_edge("dispatch_ticket", END)

        return workflow.compile()

    app = build_graph()
    print("start workflow")
    output = app.invoke(
        {"messages": ["my tv is broken"], "user_intent": None, "next_step": None}
    )
    print("agent output:", output["messages"])


def test_langgraph_cond_01():
    class UserState(TypedDict):
        age: int
        status: NotRequired[str]

    def check_age(state: UserState) -> UserState:
        print(f"[check_age] received state: {state}")
        return state

    def is_adult(state: UserState) -> bool:
        return state["age"] >= 18  # 判断是否成年

    def handle_adult(state: UserState) -> UserState:
        print("[handle_adult] user is adult")
        return UserState(age=state["age"], status="Adult")

    def handle_minor(state: UserState) -> UserState:
        print("[handle_minor] user is minor")
        return UserState(age=state["age"], status="Minor")

    def build_graph():
        builder = StateGraph(UserState)

        builder.add_node("check_age", check_age)
        builder.add_node("handle_adult", handle_adult)
        builder.add_node("handle_minor", handle_minor)

        builder.add_conditional_edges(
            "check_age",
            is_adult,
            {
                True: "handle_adult",
                False: "handle_minor",
            },
        )

        builder.add_edge(START, "check_age")
        builder.add_edge("handle_adult", END)
        builder.add_edge("handle_minor", END)

        return builder.compile()

    graph = build_graph()
    print(graph.invoke(UserState(age=20)))

    print()
    print(graph.invoke(UserState(age=15)))


def test_langgraph_loop_01():
    class ProcessState(TypedDict):
        value: int
        processed: NotRequired[bool]

    def check_value(state: ProcessState) -> ProcessState:
        print(f"[check_value] current value = {state['value']}")
        return state

    def is_positive(state: ProcessState) -> bool:
        return state["value"] > 0

    def decrement(state: ProcessState) -> ProcessState:
        result = state["value"] - 1
        print(f"[decrement] value is positive, minus 1: {result}")
        return ProcessState(value=result)

    def end_process(state: ProcessState) -> ProcessState:
        print("[end_process] value is nagtive, and exit process")
        return ProcessState(value=state["value"], processed=True)

    def build_graph():
        builder = StateGraph(ProcessState)

        builder.add_node("check_value", check_value)
        builder.add_node("decrement", decrement)
        builder.add_node("end_node", end_process)

        builder.add_conditional_edges(
            "check_value",
            is_positive,
            {
                True: "decrement",
                False: "end_node",
            },
        )

        builder.add_edge(START, "check_value")
        builder.add_edge("decrement", "check_value")
        builder.add_edge("end_node", END)

        return builder.compile()

    graph = build_graph()
    print(graph.invoke(ProcessState(value=5), config={"recursion_limit": 20}))


if __name__ == "__main__":
    # test_langgraph_seq_01()
    # test_langgraph_seq_02()

    # test_langgraph_cond_01()
    test_langgraph_loop_01()
