import os
from typing import Annotated, Callable, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from rich.console import Console
from rich.markdown import Markdown

# env

load_dotenv()

console = Console()


def setup_env():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Agentic Architecture - Multi-Agent (OpenAI)"

    for key in ["OPENAI_API_KEY", "LANGCHAIN_API_KEY"]:
        if not os.environ.get(key):
            print(f"{key} does not found. pls add in [.env] file")
    print("env and langchain tracing config done")


# llm and tools


@tool
def web_search(query: str) -> str:
    if query == "wuhan":
        return "raining"
    return "sunny"


def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.2,
    api_key=get_api_key,
    base_url="https://api.openai.com/v1",
)

llm_with_tools = llm.bind_tools([web_search])

# single agent


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def monolithic_agent_node(state: AgentState):
    console.print("--- MONOLITHIC AGENT: Thinking... ---")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:  # type: ignore
        return "tools"
    return END


def single_agent_main():
    # build agent
    tool_node = ToolNode([web_search])

    mono_graph_builder = StateGraph(AgentState)
    mono_graph_builder.add_node("agent", monolithic_agent_node)
    mono_graph_builder.add_node("tools", tool_node)

    mono_graph_builder.set_entry_point("agent")
    mono_graph_builder.add_conditional_edges("agent", should_continue)
    mono_graph_builder.add_edge("agent", "tools")

    monolithic_agent_app = mono_graph_builder.compile()
    console.print("monolithic agent graph compiled successfully")

    # run
    system_prompt = "You are a single, expert financial analyst. You must create a comprehensive report covering all aspects of the user's request in Chinese."

    company = "Zhipu AI"
    monolithic_query = f"""Create a brief but comprehensive market analysis report for {company} in Chinese.
The report should include three sections:
1. A summary of recent news and market sentiment.
2. A basic technical analysis of the stock's price trend.
3. A look at the company's recent financial performance.
"""

    final_mono_output = monolithic_agent_app.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=monolithic_query),
            ]
        }
    )

    console.print("\n--- [bold red]Final Report from Monolithic Agent[/bold red] ---")
    console.print(Markdown(final_mono_output["messages"][-1].content))


# multiple agents


class MultiAgentState(TypedDict):
    user_request: str
    news_report: Optional[str]
    technical_report: Optional[str]
    financial_report: Optional[str]
    final_report: Optional[str]


def create_specialist_node(persona: str, node_key: str) -> Callable:
    """Factory function to create a specialist agent node."""
    system_prompt = (
        persona
        + "\n\nYou have access to a web search tool. Your output MUST be a concise report section in Chinese, formatted in markdown, focusing only on your area of expertise."
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{user_request}")]
    )

    agent = prompt_template | llm_with_tools

    def specialist_node(state: MultiAgentState):
        console.print(
            f"--- CALLING {node_key.replace('_report','').upper()} ANALYST ---"
        )
        result = agent.invoke({"user_request": state["user_request"]})
        content = (
            result.content
            if result.content
            else f"No direct content, tool calls: {result.tool_calls}"
        )
        return {node_key: content}

    return specialist_node


def report_writer_node(state: MultiAgentState):
    """The manager agent that synthesizes the specialist reports."""
    console.print("--- CALLING REPORT WRITER ---")
    prompt = f"""You are an expert financial editor. Your task is to combine the following specialist reports into a single, professional, and cohesive market analysis report in Chinese. Add a brief introductory and concluding paragraph.

News & Sentiment Report:
{state['news_report']}

Technical Analysis Report:
{state['technical_report']}

Financial Performance Report:
{state['financial_report']}
"""
    final_report = llm.invoke(prompt).content
    return {"final_report": final_report}


def multiple_agents_main():
    # # build agents
    news_analyst_node = create_specialist_node(
        "You are an expert News Analyst. Your specialty is scouring the web for the latest news, articles, and social media sentiment about a company.",
        "news_report",
    )
    technical_analyst_node = create_specialist_node(
        "You are an expert Technical Analyst. You specialize in analyzing stock price charts, trends, and technical indicators.",
        "technical_report",
    )
    financial_analyst_node = create_specialist_node(
        "You are an expert Financial Analyst. You specialize in interpreting financial statements and performance metrics.",
        "financial_report",
    )

    multi_agent_graph_builder = StateGraph(MultiAgentState)
    multi_agent_graph_builder.add_node("news_analyst", news_analyst_node)
    multi_agent_graph_builder.add_node("technical_analyst", technical_analyst_node)
    multi_agent_graph_builder.add_node("financial_analyst", financial_analyst_node)
    multi_agent_graph_builder.add_node("report_writer", report_writer_node)

    multi_agent_graph_builder.set_entry_point("news_analyst")
    multi_agent_graph_builder.add_edge("news_analyst", "technical_analyst")
    multi_agent_graph_builder.add_edge("technical_analyst", "financial_analyst")
    multi_agent_graph_builder.add_edge("financial_analyst", "report_writer")
    multi_agent_graph_builder.add_edge("report_writer", END)

    multi_agent_app = multi_agent_graph_builder.compile()
    console.print("multiple agents graph compiled successfully")

    # run
    company = "Zhipu AI"
    multi_agent_query = (
        f"Create a brief but comprehensive market analysis report for {company}."
    )
    input_state = MultiAgentState(**{"user_request": multi_agent_query})
    final_multi_agent_output = multi_agent_app.invoke(input_state)

    console.print(
        "\n--- [bold green]Final Report from Multi-Agent Team[/bold green] ---"
    )
    console.print(Markdown(final_multi_agent_output["final_report"]))


if __name__ == "__main__":
    setup_env()
    # single_agent_main()
    multiple_agents_main()
