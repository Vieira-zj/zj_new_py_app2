from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

# Demo 01


class TextState(TypedDict):
    text: str


def test_langgraph_01():
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


# Demo 02


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


def test_langgraph_02():
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

    app = workflow.compile()
    print("start workflow")
    output = app.invoke(
        {"messages": ["my tv is broken"], "user_intent": None, "next_step": None}
    )
    print("agent output:", output["messages"])


if __name__ == "__main__":
    test_langgraph_01()
