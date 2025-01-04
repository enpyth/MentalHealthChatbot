""" 
run an agent without server
"""

import asyncio
from rich import print
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from IPython.display import Image
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

load_dotenv()

from agents import DEFAULT_AGENT, agents  # noqa: E402

thread_id = uuid4()
async def run_agents(agent_name: str, patient_input: str) -> None:
    agent: CompiledStateGraph = agents[agent_name]
    inputs = {
        "messages": [
            HumanMessage(content=f"{patient_input}"),
        ]
    }
    res = await agent.ainvoke(
        inputs,
        config=RunnableConfig(configurable={"thread_id": thread_id}),
    )
    # print(f"[blue]{res['messages']}[/blue]")
    res["messages"][-1].pretty_print()


def display_mermaid(agent_name: str, path: str):
    """create arch img"""
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d")

    agent: CompiledStateGraph = agents[agent_name]
    print(agent.get_graph().draw_mermaid())
    graph_img = Image(
        agent.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
    with open(f"{path}{agent_name}_graph_{formatted_time}.png", "wb") as f:
        f.write(graph_img.data)


def conversation(agent_name: str):
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        asyncio.run(run_agents(agent_name, query))


if __name__ == "__main__":
    TEST_NAME = "demo_agent"
    # conversation(TEST_NAME)
    display_mermaid(TEST_NAME, "../imgs/")
