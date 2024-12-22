from langgraph.graph.state import CompiledStateGraph

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.research_assistant import research_assistant
from agents.counsellor import counsellor
from agents.demo_agent import demo_agent
DEFAULT_AGENT = "demo_agent"


agents: dict[str, CompiledStateGraph] = {
    "chatbot": chatbot,
    "research-assistant": research_assistant,
    "bg-task-agent": bg_task_agent,
    "counsellor": counsellor,
    "demo_agent": demo_agent,
}
