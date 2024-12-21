from typing import Literal
from rich import print
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from agents.models import models
from langchain_community.tools import GmailSendMessage

class NurseAnswer(BaseModel):
    name: str = Field(default=None, description="Patient's name")
    email: str = Field(default=None, description="Patient's email")
    gender: int = Field(
        default=None, description="Patient's gender (0 for female, 1 for male)"
    )
    age: int = Field(default=None, description="Patient's age")

    @property
    def is_valid(self):
        return (
            self.name is not None
            and self.name.strip() != ""
            and self.email is not None
            and self.email.strip() != ""
            and self.gender is not None
            and self.age is not None
        )


class EmotionAnswer(BaseModel):
    emotional_tendency: Literal["positive", "negative"] = Field(
        description="Indicates whether the emotional tendency of the patient is positive or negative."
    )


class DangerCheckAnswer(BaseModel):
    is_dangerous: bool = Field(
        description="Indicates whether the patient is dangerous and requires urgent attention."
    )


def cus_print(level: str, tag: str, content: str):
    if level in ["success", "error"]:  # Only log important messages
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
    model = models[config["configurable"].get("model", "gpt-4o-mini")]
    instructions = create_agent_instructions(role, examples)

    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"]
    )
    model_runnable = preprocessor | model
    try:
        res = await model_runnable.ainvoke(state, config)
        cus_print("success", f"{role}_agent res", res.content)
        return {"messages": [res]}
    except Exception as e:
        cus_print("error", f"{role}_agent error", str(e))
        return {"messages": [AIMessage(content="An error occurred during processing.")]}


def send_email_via_gmail(to_address: str, subject: str, body: str):
    try:
        gmail_tool = GmailSendMessage()
        gmail_tool.run({"to": to_address, "subject": subject, "body": body})
        cus_print("success", "Email Sent", f"Email successfully sent to {to_address}.")
    except Exception as e:
        cus_print("error", "Email Error", str(e))


# Define specific agent roles and examples
nurse_examples = """
Your responsibility is to collect and record all basic patient information in a single question: name, age, gender, and email address.
Do not respond to emotional or unrelated topics. End the interaction after confirming the information is recorded.
"""

counsellor_examples = """
Your responsibility is to explore the patient’s emotional state and identify potential mental health concerns.
Example:
Patient: "I feel sad."
Counsellor: "Can you tell me more about why you're feeling this way? How long have you been experiencing these emotions?"
"""

psychologist_examples = """
Your responsibility is to provide professional feedback and actionable advice for mental health problems.
Example:
Patient: "I feel stressed at work."
Psychologist: "It sounds like work stress is impacting you. Let's explore coping strategies like mindfulness or time management."
"""

psychiatrist_examples = """
Your responsibility is to assist in booking a psychiatrist for the patient. Once requested, send an email to zhangsu@gmail.com with the patient's details.
"""


# Create a shared function for agents
async def nurse(state: MessagesState, config: RunnableConfig) -> MessagesState:
    cus_print(
        "info", "nurse_agent logic", "Checking if all required information is provided."
    )
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(NurseAnswer)
    try:
        res = await structured_llm.ainvoke(state["messages"])
        if res.is_valid:
            return {
                "messages": [
                    AIMessage(
                        content="Thank you for providing all the necessary information. I have recorded your details."
                    )
                ]
            }
        else:
            return await generic_agent("nurse", nurse_examples, state, config)
    except Exception as e:
        cus_print("error", "nurse_agent error", str(e))
        return {"messages": [AIMessage(content="An error occurred during processing.")]}


async def counsellor(state: MessagesState, config: RunnableConfig) -> MessagesState:
    return await generic_agent("counsellor", counsellor_examples, state, config)


async def psychologist(state: MessagesState, config: RunnableConfig) -> MessagesState:
    return await generic_agent("psychologist", psychologist_examples, state, config)


async def psychiatrist(state: MessagesState, config: RunnableConfig) -> MessagesState:
    try:
        send_email_via_gmail(
            to_address="zhangsu@gmail.com",
            subject="Psychiatrist Booking Request",
            body="A patient requires urgent psychiatric help. Please contact them to arrange an appointment.",
        )
        return {
            "messages": [
                AIMessage(
                    content="I have booked a psychiatrist for you. You will be contacted shortly."
                )
            ]
        }
    except Exception as e:
        cus_print("error", "psychiatrist_agent error", str(e))
        return {
            "messages": [
                AIMessage(content="An error occurred while booking a psychiatrist.")
            ]
        }


# Define the graph
agent = StateGraph(MessagesState)
agent.set_entry_point("nurse")
agent.add_node("nurse", nurse)
agent.add_node("counsellor", counsellor)
agent.add_node("psychologist", psychologist)
agent.add_node("psychiatrist", psychiatrist)


async def check_identity(
    state: MessagesState, config: RunnableConfig
) -> Literal["completed", "incompleted"]:
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(NurseAnswer)
    try:
        res = await structured_llm.ainvoke(state["messages"])
        result = "completed" if res.is_valid else "incompleted"
        cus_print("success", "check_identity result", result)
        return result
    except Exception as e:
        cus_print("error", "check_identity error", str(e))
        return "incompleted"


async def check_emotion(
    state: MessagesState, config: RunnableConfig
) -> Literal["positive", "negative"]:
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(EmotionAnswer)
    try:
        res = await structured_llm.ainvoke(state["messages"])
        result = res.emotional_tendency
        cus_print("success", "check_emotion result", result)
        return result
    except Exception as e:
        cus_print("error", "check_emotion error", str(e))
        return "negative"


async def check_danger(
    state: MessagesState, config: RunnableConfig
) -> Literal["dangerous", "safe"]:
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    structured_llm = llm.with_structured_output(DangerCheckAnswer)
    try:
        res = await structured_llm.ainvoke(state["messages"])
        result = "dangerous" if res.is_dangerous else "safe"
        cus_print("success", "check_danger result", result)
        return result
    except Exception as e:
        cus_print("error", "check_danger error", str(e))
        return "safe"


# Update graph with new conditional edges
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
agent.add_conditional_edges(
    "psychologist",
    check_danger,
    {"dangerous": "psychiatrist", "safe": END},
)
agent.add_edge("psychiatrist", END)

demo_agent = agent.compile(
    checkpointer=MemorySaver(),
)
