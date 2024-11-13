import asyncio
import os
import streamlit as st

APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

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
        WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. I may take a few seconds to boot up when you send your first message. Ask me anything!"
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
            )
            await draw_messages(stream, is_new=True)
        else:
            response = await agent_client.ainvoke(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun()  # Clear stale containers

    # If messages have been generated, show feedback widget
    if len(messages) > 0:
        with st.session_state.last_message:
            await handle_feedback()


if __name__ == "__main__":
    asyncio.run(main())
