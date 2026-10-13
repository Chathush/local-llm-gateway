"""A complete LangGraph tool loop; the gateway only makes model decisions."""

import argparse
import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from local_llm_gateway.integrations.langchain import GatewayChatModel


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def build_graph(model):
    bound = model.bind_tools([add])

    async def call_model(state):
        result = await bound.ainvoke(state["messages"])
        return {"messages": [result]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode([add]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


async def main(profile):
    model = GatewayChatModel.from_config(profile=profile)
    result = await build_graph(model).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="You must use the add tool to compute 17 + 25, then give the answer."
                )
            ]
        },
        config={"recursion_limit": 8},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="primary")
    asyncio.run(main(parser.parse_args().profile))
