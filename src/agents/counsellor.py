import os
from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.models import models
from agents.tools import calculator


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    is_last_step: IsLastStep


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if os.getenv("OPENWEATHERMAP_API_KEY") is not None:
    tools.append(OpenWeatherMapQueryRun(name="Weather"))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


# reference https://python.langchain.com/docs/how_to/functions/
def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def emotion_analysis(state: AgentState) -> AgentState:
    content = "TODO"
    return AIMessage(content=content)

async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


async def nurse(state: AgentState) -> AgentState:
    content = "TODO"
    return AIMessage(content=content)


async def ethical_check(state: AgentState) -> AgentState:
    content = "TODO"
    return AIMessage(content=content)


async def psychologist(state: AgentState) -> AgentState:
    content = "TODO"
    return AIMessage(content=content)


async def psychiatrist(state: AgentState) -> AgentState:
    content = "TODO"
    return AIMessage(content=content)


# start asking if info is completed
def edge_nurse():
    pass


def edge_diagnose():
    pass


def rag():
    pass


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


# After "model", if there are tool calls, run "tools". Otherwise END.
def edge_counsellor(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "ethical_check"


# Define the graph
agent = StateGraph(AgentState)
agent.set_entry_point("nurse")
agent.add_node("nurse", nurse)
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("counsellor", acall_model)
agent.add_node("emotion_analysis", emotion_analysis)
agent.add_node("tools", ToolNode(tools))
agent.add_node("ethical_check", ethical_check)
agent.add_node("psychologist", psychologist)
agent.add_node("rag", rag)
agent.add_node("psychiatrist", psychiatrist)

agent.add_conditional_edges(
    "nurse",
    edge_nurse,
    {"info_completed": "guard_input", "info_incompleted": END},
)


agent.add_conditional_edges(
    "guard_input",
    check_safety,
    {"unsafe": "block_unsafe_content", "safe": "counsellor"},
)

agent.add_edge("block_unsafe_content", END)
agent.add_conditional_edges(
    "counsellor",
    edge_counsellor,
    {
        "tools": "tools",
        "emotion_analysis": "emotion_analysis",
    },
)
agent.add_conditional_edges(
    "emotion_analysis",
    edge_counsellor,
    {
        "info": "ethical_check",
        "warn": "psychologist",
    },
)
agent.add_edge("tools", "counsellor")
agent.add_conditional_edges(
    "psychologist",
    edge_diagnose,
    {
        "fatal": "psychiatrist",
        "medical_advice": "rag",
    },
)

agent.add_edge("rag", "ethical_check")
agent.add_edge("psychiatrist", "ethical_check")
agent.add_edge("ethical_check", END)

counsellor = agent.compile(
    checkpointer=MemorySaver(),
)
