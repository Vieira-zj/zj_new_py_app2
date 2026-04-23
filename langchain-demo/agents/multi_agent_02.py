from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

# ----------------------------
# 1. State
# ----------------------------


class UserInfo(TypedDict, total=False):
    name: str
    loyalty_id: str
    preferences: Dict[str, Any]


class TravelState(TypedDict):
    messages: List[Any]
    user_info: UserInfo
    agent_stack: List[Literal["main", "flights", "hotels", "cars", "visa"]]
    pending_action: Optional[Dict[str, Any]]


# ----------------------------
# 2. Tools
# ----------------------------


@tool
def search_flights(origin: str, destination: str, date: str) -> Dict[str, Any]:
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "options": [
            {"flight_no": "MU123", "depart": "09:10", "arrive": "13:40", "price": 2100},
            {"flight_no": "NH008", "depart": "12:00", "arrive": "16:30", "price": 2850},
        ],
    }


@tool
def search_hotels(city: str, checkin: str, checkout: str) -> Dict[str, Any]:
    return {
        "city": city,
        "checkin": checkin,
        "checkout": checkout,
        "options": [
            {"hotel": "Shinjuku Stay", "price_per_night": 980, "refundable": True},
            {"hotel": "Ginza Business", "price_per_night": 1180, "refundable": False},
        ],
    }


@tool
def cancel_hotel(order_id: str) -> Dict[str, Any]:
    return {"order_id": order_id, "status": "cancelled", "refund": 1200}


@tool
def update_ticket(ticket_id: str, new_flight_no: str) -> Dict[str, Any]:
    return {
        "ticket_id": ticket_id,
        "new_flight_no": new_flight_no,
        "status": "rebooked",
    }


SAFE_TOOLS = [search_flights, search_hotels]
SENSITIVE_TOOLS = [cancel_hotel, update_ticket]


# ----------------------------
# 3. Tool Router
# ----------------------------


@tool
def ToFlightAssistant() -> str:
    return "route:flights"


@tool
def ToHotelAssistant() -> str:
    return "route:hotels"


ROUTING_TOOLS = [ToFlightAssistant, ToHotelAssistant]


def route_from_main(state: TravelState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        tool_name = last.tool_calls[0]["name"]
        if tool_name == "ToFlightAssistant":
            return "to_flights"
        if tool_name == "ToHotelAssistant":
            return "to_hotels"
    return "end_or_wait"


# ----------------------------
# 4. LLM
# ----------------------------


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )


# ----------------------------
# 5. Entry Node
# ----------------------------


def create_entry_node(target: Literal["flights", "hotels"]) -> Any:
    def _entry(state: TravelState) -> TravelState:
        stack = state["agent_stack"]
        stack.append(target)
        background = (
            "你是旅行系统里的专业助手.\n"
            f"你现在负责: {target}.\n"
            "你会收到主助手已有的对话上下文, 请直接推进任务.\n"
            "如果任务完成, 返回 'CompleteOrEscalate' 工具并说明原因.\n"
            "如果需要执行敏感操作 (取消/改签), 先触发授权流程, 不要擅自执行."
        )
        return {
            **state,
            "messages": [SystemMessage(content=background)] + state["messages"],
            "agent_stack": stack,
        }

    return _entry


@tool
def CompleteOrEscalate(reason: str) -> str:
    return f"complete: {reason}"


# ----------------------------
# 6. Interrupt Router
# ----------------------------


def route_after_child(state: TravelState) -> str:
    # 如果有待授权敏感操作, 先走网关
    if state.get("pending_action"):
        return "sensitive_gateway"
    # 如果子助手显式 CompleteOrEscalate, 则返回主助手
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            if tc["name"] == "CompleteOrEscalate":
                stack = state["agent_stack"]
                if len(stack) > 1:
                    stack.pop()
                state["agent_stack"] = stack
                return "back_to_main"
    # 否则继续留在当前子助手
    return "stay"


def route_after_gateway(state: TravelState) -> str:
    top = state["agent_stack"][-1]
    return {"main": "main", "flights": "flights", "hotels": "hotels"}.get(top, "main")


def sensitive_tool_gateway(state: TravelState) -> TravelState:
    """
    如果 pending_action 存在, 则触发 interrupt 请求用户授权. 用户同意后继续执行; 用户拒绝则清空并返回上级.
    """
    pending = state.get("pending_action")
    if not pending:
        return state

    decision = interrupt(
        {
            "type": "approval_required",
            "title": "需要确认敏感操作",
            "action": pending,
            "prompt": "请回复: yes/no",
        }
    )
    # approved
    if isinstance(decision, dict) and decision.get("approved") is True:
        tool_name = pending["tool"]
        args = pending["args"]
        # exec tool
        if tool_name == "cancel_hotel":
            result = cancel_hotel.invoke(args)
        elif tool_name == "update_ticket":
            result = update_ticket.invoke(args)
        else:
            result = {"error": f"unknown sensitive tool: {tool_name}"}
        return {
            **state,
            "pending_action": None,
            "messages": state["messages"]
            + [ToolMessage(content=str(result), tool_call_id=pending["tool_call_id"])],
        }
    # rejected: 清空 pending_action, 并回到主助手
    stack = state["agent_stack"]
    if len(stack) > 1:
        stack.pop()
    return {
        **state,
        "pending_action": None,
        "agent_stack": stack,
        "messages": state["messages"]
        + [AIMessage(content="好的, 我不会执行该操作. 我们回到主流程继续.")],
    }


# ----------------------------
# 7. Agents
# ----------------------------


def main_assistant(state: TravelState) -> TravelState:
    llm = create_llm().bind_tools(ROUTING_TOOLS)
    sys = SystemMessage(
        content=(
            "你是旅行规划系统的主助手 (调度中心).\n"
            "职责: 识别用户意图; 必要时委派到子助手 (flight/hotel); 简单问题直接回答.\n"
            "当用户提出需要取消/改签等修改动作时, 引导到对应子助手处理, 并强调需要用户确认.\n"
        )
    )
    msgs = [sys] + state["messages"]
    out = llm.invoke(msgs)
    return {**state, "messages": state["messages"] + [out]}


def flight_assistant(state: TravelState) -> TravelState:
    sys = SystemMessage(
        content=(
            "你是航班助手.\n"
            "你可以用 search_flights 查询; 如果要改签 (update_ticket) 或退票等敏感操作, 必须先走授权.\n"
            "当你准备执行敏感工具时, 不要直接调用工具; 请在 state.pending_action 写入待执行信息, 并让系统走授权网关.\n"
        )
    )
    llm = create_llm().bind_tools([*SAFE_TOOLS, *SENSITIVE_TOOLS, CompleteOrEscalate])
    out = llm.invoke([sys] + state["messages"])

    if hasattr(out, "tool_calls") and out.tool_calls:
        for tc in out.tool_calls:
            name = tc["name"]
            if name in {"update_ticket"}:
                return {
                    **state,
                    "messages": state["messages"] + [out],
                    "pending_action": {
                        "tool": name,
                        "args": tc["args"],
                        "tool_call_id": tc["id"],
                    },
                }
    return {**state, "messages": state["messages"] + [out]}


def hotel_assistant(state: TravelState) -> TravelState:
    sys = SystemMessage(
        content=(
            "你是酒店助手.\n"
            "你可以用 search_hotels 查询; 如果要取消订单 (cancel_hotel) 等敏感操作, 必须先走授权.\n"
            "当你准备执行敏感工具时, 不要直接调用工具; 请在 state.pending_action 写入待执行信息, 并让系统走授权网关.\n"
        )
    )
    llm = create_llm().bind_tools([*SAFE_TOOLS, *SENSITIVE_TOOLS, CompleteOrEscalate])
    out = llm.invoke([sys] + state["messages"])

    if hasattr(out, "tool_calls") and out.tool_calls:
        for tc in out.tool_calls:
            name = tc["name"]
            if name in {"cancel_hotel"}:
                return {
                    **state,
                    "messages": state["messages"] + [out],
                    "pending_action": {
                        "tool": name,
                        "args": tc["args"],
                        "tool_call_id": tc["id"],
                    },
                }
    return {**state, "messages": state["messages"] + [out]}


# ----------------------------
# 8. Graph
# ----------------------------


def build_graph() -> Any:
    g = StateGraph(TravelState)
    g.add_node("main", main_assistant)
    g.add_node("entry_flights", create_entry_node("flights"))
    g.add_node("entry_hotels", create_entry_node("hotels"))
    g.add_node("flights", flight_assistant)
    g.add_node("hotels", hotel_assistant)
    g.add_node("sensitive_gateway", sensitive_tool_gateway)

    g.set_entry_point("main")
    g.add_conditional_edges(
        "main",
        route_from_main,
        {
            "to_flights": "entry_flights",
            "to_hotels": "entry_hotels",
            "end_or_wait": END,
        },
    )
    g.add_edge("entry_flights", "flights")
    g.add_edge("entry_hotels", "hotels")
    g.add_conditional_edges(
        "flights",
        route_after_child,
        {
            "sensitive_gateway": "sensitive_gateway",
            "back_to_main": "main",
            "stay": "flights",
        },
    )
    g.add_conditional_edges(
        "hotels",
        route_after_child,
        {
            "sensitive_gateway": "sensitive_gateway",
            "back_to_main": "main",
            "stay": "hotels",
        },
    )
    g.add_conditional_edges(
        "sensitive_gateway",
        route_after_gateway,
        {"main": "main", "flights": "flights", "hotels": "hotels"},
    )
    return g.compile()


# ----------------------------
# 9. Main
# ----------------------------


def main():
    graph = build_graph()
    state: TravelState = {
        "messages": [],
        "user_info": {"name": "Alex", "preferences": {"hotel_budget": 1200}},
        "agent_stack": ["main"],
        "pending_action": None,
    }
    print("输入 exit 退出")

    while True:
        text = input("\npls input:").strip()
        if text.lower() == "exit":
            break
        state["messages"].append(HumanMessage(content=text))
        # invoke 可能触发 interrupt, 这里用 try/except 演示一个 简化版人工审批
        try:
            state = graph.invoke(state)
        except Exception as e:
            print(f"\n[interrupt] {e}")
            pending = state.get("pending_action")
            if pending:
                print(f"待执行: {pending['tool']} args={pending['args']}")
                decision = input("确认执行? (yes/no)>").strip()
                approved = decision == "yes"
                state["messages"].append(
                    HumanMessage(content=f"审批结果: {'yes' if approved else 'no'}")
                )
                state = sensitive_tool_gateway({**state, "pending_action": pending})
                state = graph.invoke(state)
        # 打印最后一条 AI 输出
        for m in state["messages"][-3:]:
            if isinstance(m, AIMessage):
                print(f"\n助手: {m.content}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")

    main()
