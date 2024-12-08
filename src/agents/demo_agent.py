from typing import Literal
from rich import print
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate

from agents.models import models
from agents.check import NurseAnswer, EmotionAnswer


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


def cus_print(level: str, tag: str, content: str):
    colors: dict = {
        "info": "blue",
        "success": "green",
        "warn": "dark_orange",
        "error": "red",
    }
    color: str = colors.get(level)
    output: str = f"[{color}][bold]{tag}[/bold]: {content} [{color}]"
    print(output)


async def nurse(state: AgentState, config: RunnableConfig) -> AgentState:
    cus_print("info", "nurse_agent begin", str(state["messages"]))
    model = models[config["configurable"].get("model", "gpt-4o-mini")]
    instructions = "You are a nurse, your responsibility is recording patient's information including name, age, gender, and email address."
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    model_runnable = preprocessor | model
    res = await model_runnable.ainvoke(state, config)
    cus_print("success", "nurse_agent end", res.content)
    return {"messages": [res]}


async def counsellor(state: AgentState, config: RunnableConfig) -> AgentState:
    cus_print("info", "counsellor_agent begin", str(state["messages"]))
    model = models[config["configurable"].get("model", "gpt-4o-mini")]
    instructions = "You are a counsellor of mental health. Your responsibility is talk with patients to find potential mental problems and provide useful information to professional psychologist."
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
    )
    model_runnable = preprocessor | model
    res = await model_runnable.ainvoke(state, config)
    cus_print("success", "counsellor_agent end", res.content)
    # print(f"[green]counsellor_agent, res: {res}[/green]")
    return {"messages": [res]}


async def psychologist(state: AgentState, config: RunnableConfig) -> AgentState:
    cus_print("info", "psychologist_agent begin", str(state["messages"]))
    model = models[config["configurable"].get("model", "gpt-4o-mini")]
    instructions = "You are a professional psychologist. Patients who may suffer from mental health problems need your feedback and advices. "
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
    )
    model_runnable = preprocessor | model
    res = await model_runnable.ainvoke(state, config)
    cus_print("success", "psychologist_agent end", res.content)
    # print(f"[green]psychologist_agent, res: {res}[/green]")
    return {"messages": [res]}


# Define the graph
agent = StateGraph(AgentState)
agent.set_entry_point("nurse")
agent.add_node("nurse", nurse)
agent.add_node("counsellor", counsellor)
agent.add_node("psychologist", psychologist)


async def check_identity(
    state: AgentState, config: RunnableConfig
) -> Literal["completed", "incompleted"]:
    cus_print("info", "check_identity begin", str(state["messages"]))

    # LLM with function call
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(NurseAnswer)
    res = await structured_llm.ainvoke(state["messages"])

    if res.is_completed:
        cus_print("warn", "check_identity end, edge to", res.is_completed)
        return "completed"
    else:
        cus_print("warn", "check_identity end, edge to", res.is_completed)
        return "incompleted"


async def check_emotion(
    state: AgentState, config: RunnableConfig
) -> Literal["positive", "negative"]:
    cus_print("info", "check_emotion begin", str(state["messages"]))

    # LLM with function call
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(EmotionAnswer)
    res = await structured_llm.ainvoke(state["messages"])

    cus_print("warn", "check_emotion end, edge to", res.emotional_tendency)
    return res.emotional_tendency


# Always END after blocking unsafe content
agent.add_conditional_edges(
    "nurse",
    check_identity,
    {"completed": "counsellor", "incompleted": END},
)
agent.add_conditional_edges(
    "counsellor",
    check_emotion,
    {"positive": END, "negative": "psychologist"},
)
agent.add_edge("psychologist", END)

demo_agent = agent.compile(
    checkpointer=MemorySaver(),
)
