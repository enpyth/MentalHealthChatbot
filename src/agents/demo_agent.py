from typing import Literal
from rich import print
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from agents.models import models


class NurseAnswer(BaseModel):
    name: str = Field(default=None, description="Patient's name")
    email: str = Field(default=None, description="Patient's email")
    gender: int = Field(
        default=None, description="Patient's gender (0 for female, 1 for male)"
    )
    age: int = Field(default=None, description="Patient's age")

    @property
    def is_valid(self):
        result = (
            self.name is not None and self.name.strip() != "" and
            self.email is not None and self.email.strip() != "" and
            self.gender is not None and
            self.age is not None
        )
        print(f"Validation Results: name={self.name}, email={self.email}, gender={self.gender}, age={self.age}, is_valid={result}")
        return result


class EmotionAnswer(BaseModel):
    emotional_tendency: Literal["positive", "negative"] = Field(
        description="Indicates whether the emotional tendency of the patient is positive or negative."
    )


def cus_print(level: str, tag: str, content: str):
    colors: dict = {
        "info": "blue",
        "success": "green",
        "warn": "dark_orange",
        "error": "red",
    }
    color: str = colors.get(level, "blue")
    output: str = f"[{color}][bold]{tag}[/bold]: {content} [{color}]"
    print(output)


def create_agent_instructions(role: str, examples: str) -> str:
    return f"""
    You are a {role}. {examples}
    """


async def generic_agent(
    role: str, examples: str, state: MessagesState, config: RunnableConfig
) -> MessagesState:
    cus_print("info", f"{role}_agent begin", str(state.get("messages", [])))
    model = models[config["configurable"].get("model", "gpt-4o-mini")]
    instructions = create_agent_instructions(role, examples)

    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"]
    )
    model_runnable = preprocessor | model
    try:
        res = await model_runnable.ainvoke(state, config)
        cus_print("success", f"{role}_agent end", res.content)
        return {"messages": [res]}
    except Exception as e:
        cus_print("error", f"{role}_agent error", str(e))
        return {"messages": [AIMessage(content="An error occurred during processing.")]}


# Define specific agent roles and examples
nurse_examples = """
Your responsibility is to collect and record all basic patient information in a single question: name, age, gender, and email address.
Example interactions:
Patient: "Hello"
Nurse: "Hi, I am here to collect your basic information. Could you please provide your name, age, gender, and email address?"
"""

counsellor_examples = """
Your responsibility is to focus on understanding the patient’s mental health and identify potential problems. Do not collect basic information or provide professional advice.
Example interactions:
Patient: "I feel sad."
Counsellor: "Can you tell me more about what is bothering you?"
Patient: "I think I have anxiety."
Counsellor: "Describe your feelings so I can guide you better and provide information to a psychologist."
"""

psychologist_examples = """
Your responsibility is to provide professional feedback and advice for mental health problems. Do not ask about basic information or general emotional states.
Example interactions:
Patient: "I have been feeling very down lately."
Psychologist: "Can you describe more about your experience?"
Patient: "I don’t know what to do about my stress."
Psychologist: "Let’s explore some coping strategies."
"""


# Create a shared function for agents
async def nurse(state: MessagesState, config: RunnableConfig) -> MessagesState:
    return await generic_agent("nurse", nurse_examples, state, config)


async def counsellor(state: MessagesState, config: RunnableConfig) -> MessagesState:
    return await generic_agent("counsellor", counsellor_examples, state, config)


async def psychologist(state: MessagesState, config: RunnableConfig) -> MessagesState:
    return await generic_agent("psychologist", psychologist_examples, state, config)


# Define the graph
agent = StateGraph(MessagesState)
agent.set_entry_point("nurse")
agent.add_node("nurse", nurse)
agent.add_node("counsellor", counsellor)
agent.add_node("psychologist", psychologist)


async def check_identity(
    state: MessagesState, config: RunnableConfig
) -> Literal["completed", "incompleted"]:
    cus_print("info", "check_identity begin", str(state.get("messages", [])))
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(NurseAnswer)
    try:
        res = await structured_llm.ainvoke(state["messages"])
        return "completed" if res.is_valid else "incompleted"
    except Exception as e:
        cus_print("error", "check_identity error", str(e))
        return "incompleted"


async def check_emotion(
    state: MessagesState, config: RunnableConfig
) -> Literal["positive", "negative"]:
    cus_print("info", "check_emotion begin", str(state.get("messages", [])))
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(EmotionAnswer)
    try:
        res = await structured_llm.ainvoke(state["messages"])
        return res.emotional_tendency
    except Exception as e:
        cus_print("error", "check_emotion error", str(e))
        return "negative"


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
