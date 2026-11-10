import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Let the .env value replace any old PowerShell value.
load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from the .env file.")


# Streamlit reruns the script on every interaction, so create the model once.
@st.cache_resource
def get_model():
    return init_chat_model(
        "openai/gpt-oss-20b",
        model_provider="groq",
        api_key=api_key,
        temperature=0.7,
    )


model = get_model()

st.title("Chatbot")

# Keep the conversation history across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]

# Show previous messages (skip the system message).
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

prompt = st.chat_input("Type your message...")

if prompt and prompt.strip():
    prompt = prompt.strip()

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    response = model.invoke(st.session_state.messages)

    with st.chat_message("assistant"):
        st.write(response.content)
    st.session_state.messages.append(AIMessage(content=response.content))