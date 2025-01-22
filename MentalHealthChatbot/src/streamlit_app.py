import asyncio
import os
from collections.abc import AsyncGenerator
import sqlite3
from hashlib import sha256
from datetime import datetime
import uuid

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient
from schema import ChatHistory, ChatMessage
from schema.task_data import TaskData
from utils.db import (
    init_db,
    authenticate_user,
    create_user,
    create_conversation,
    get_user_conversations,
    update_conversation_timestamp
)

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"


async def main() -> None:
    # Initialize session state variables
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Show authentication interface if not authenticated
    if not st.session_state.authenticated:
        init_db()
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        with tab1:
            st.header("Sign In")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Sign In"):
                if user := authenticate_user(email, password):
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error("Invalid email or password")
        
        with tab2:
            st.header("Sign Up")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            
            if st.button("Sign Up"):
                if create_user(new_email, new_password):
                    st.success("Account created successfully! Please sign in.")
                else:
                    st.error("Email already exists")
        
        # Stop here if not authenticated
        st.stop()

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        agent_url = os.getenv("AGENT_URL", "http://localhost")
        st.session_state.agent_client = AgentClient(agent_url)
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = str(uuid.uuid4())
        thread_id = create_conversation(st.session_state.user_email, thread_id)
        messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    models = {
        "OpenAI GPT-4o-mini (streaming)": "gpt-4o-mini",
        "Gemini 1.5 Flash (streaming)": "gemini-1.5-flash",
        "Claude 3 Haiku (streaming)": "claude-3-haiku",
        "llama-3.1-70b on Groq": "llama-3.1-70b",
        "AWS Bedrock Haiku (streaming)": "bedrock-haiku",
    }
    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        with st.popover(":material/settings: Settings", use_container_width=True):
            m = st.radio("LLM to use", options=models.keys())
            model = models[m]
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=[
                    "demo_agent",
                    "research-assistant",
                    "chatbot",
                    "bg-task-agent",
                ],
            )
            use_streaming = st.toggle("Stream results", value=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        st.markdown(
            f"Thread ID: **{st.session_state.thread_id}**",
            help=f"Set URL query parameter ?thread_id={st.session_state.thread_id} to continue this conversation",
        )

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI-powered mental health assistant. Could you please provide me with your name, age, gender, and email address for the record?"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        if use_streaming:
            stream = agent_client.astream(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
                user_id=st.session_state.user_email,
            )
            await draw_messages(stream, is_new=True)
        else:
            response = await agent_client.ainvoke(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
                user_id=st.session_state.user_email,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun()  # Clear stale containers

        # Update the conversation timestamp after new messages
        update_conversation_timestamp(st.session_state.thread_id)

    # If messages have been generated, show feedback widget
    if len(messages) > 0:
        with st.session_state.last_message:
            await handle_feedback()

    if st.session_state.authenticated:
        with st.sidebar:
            col1, col2 = st.columns([4, 1])
            with col1:
                new_chat_title = st.text_input("New chat title", 
                                             value="New Conversation",
                                             key="new_chat_title",
                                             label_visibility="collapsed")
            with col2:
                if st.button("➕", help="Start new chat", use_container_width=True):
                    # Generate new thread ID using UUID
                    thread_id = str(uuid.uuid4())
                    # Create new conversation with custom title
                    thread_id = create_conversation(st.session_state.user_email, 
                                                   thread_id, 
                                                   title=new_chat_title)
                    # Reset session state
                    st.session_state.thread_id = thread_id
                    st.session_state.messages = []
                    st.rerun()

            st.divider()
            
            # Show existing conversations
            st.subheader("Your Conversations")
            conversations = get_user_conversations(st.session_state.user_email)
            
            if not conversations:
                st.info("No conversations yet. Start a new chat!")
            
            for conv_id, title, created, updated in conversations:
                col1, col2 = st.columns([5, 1])
                with col1:
                    # Format the date for display
                    updated_date = datetime.fromisoformat(updated).strftime("%Y-%m-%d %H:%M")
                    if st.button(f"📝 {title}\n{updated_date}", 
                               key=f"conv_{conv_id}", 
                               use_container_width=True):
                        st.session_state.thread_id = conv_id
                        try:
                            history: ChatHistory = agent_client.get_history(thread_id=conv_id)
                            st.session_state.messages = history.messages
                        except Exception as e:
                            # If history doesn't exist, initialize empty messages
                            st.session_state.messages = []
                            st.info("Starting a new conversation thread.")
                        st.rerun()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # This is an implementation of the TaskData example for CustomData.
                # An agent can write a CustomData object to the message stream, and
                # it's passed to the client for rendering. To see this in practice,
                # run the app with the `bg-task-agent` agent.

                # This is provided as an example, you may want to write your own
                # CustomData types and handlers. This section will be skipped for
                # any other agents that don't send CustomData.
                task_data = TaskData.model_validate(msg.custom_data)

                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not Task, create a new chat message
                # and container for task messages
                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = st.status("")
                    current_task_data: dict[str, TaskData] = {}

                status_str = f"Task **{task_data.name}** "
                match task_data.state:
                    case "new":
                        status_str += "has :blue[started]. Input:"
                    case "running":
                        status_str += "wrote:"
                    case "complete":
                        if task_data.result == "success":
                            status_str += ":green[completed successfully]. Output:"
                        else:
                            status_str += ":red[ended with error]. Output:"
                status.write(status_str)
                status.write(task_data.data)
                status.write("---")
                if task_data.run_id not in current_task_data:
                    # Status label always shows the last newly started task
                    status.update(label=f"""Task: {task_data.name}""")
                current_task_data[task_data.run_id] = task_data
                # Status is "running" until all tasks have completed
                if not any(entry.completed() for entry in current_task_data.values()):
                    state = "running"
                # Status is "error" if any task has errored
                elif any(entry.completed_with_error() for entry in current_task_data.values()):
                    state = "error"
                # Status is "complete" if all tasks have completed successfully
                else:
                    state = "complete"
                status.update(state=state)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs={"comment": "In-line human feedback"},
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
