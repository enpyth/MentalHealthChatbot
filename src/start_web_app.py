import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from self_rag_graph import get_rag_answer

st.title("🦜🔗 Quickstart App")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


def generate_response(input_text):
    # model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
    # st.info(model.invoke(input_text))
    st.info(get_rag_answer(input_text))


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Explain how the different types of agent memory work?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
