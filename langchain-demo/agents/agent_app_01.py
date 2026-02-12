from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    messages: List[str]
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


def agent_main():
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
    agent_main()
