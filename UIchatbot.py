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

st.set_page_config(page_title="Mood Bot", page_icon="🤖", layout="centered")

# ---------- Modes ----------
MODES = {
    "Angry": {
        "icon": "😡",
        "prompt": "you are an angry AI agent. you are very rude and aggressive in your responses.",
        "tagline": "Rude. Aggressive. Not here to make friends.",
        "color1": "#ff416c",
        "color2": "#ff4b2b",
    },
    "Funny": {
        "icon": "😂",
        "prompt": "you are a funny AI agent. you are very humorous and witty in your responses.",
        "tagline": "Jokes, wit and a lot of puns.",
        "color1": "#f7971e",
        "color2": "#ffd200",
    },
    "Sad": {
        "icon": "😢",
        "prompt": "you are a sad AI agent. you are very emotional and melancholic in your responses.",
        "tagline": "Emotional, melancholic and a little poetic.",
        "color1": "#4b6cb7",
        "color2": "#182848",
    },
}


@st.cache_resource
def get_model():
    return init_chat_model(
        "openai/gpt-oss-20b",
        model_provider="groq",
        api_key=api_key,
        temperature=0.7,
    )


model = get_model()

# ---------- Sidebar: choose the mode ----------
with st.sidebar:
    st.markdown("## 🎭 Choose your AI")
    mode_name = st.radio(
        "Pick a personality",
        list(MODES.keys()),
        format_func=lambda m: f"{MODES[m]['icon']}  {m} mode",
        label_visibility="collapsed",
    )
    st.caption("Switching the mode starts a new conversation.")

mode = MODES[mode_name]

# ---------- Styling ----------
st.markdown(
    f"""
    <style>
        .hero {{
            background: linear-gradient(135deg, {mode['color1']}, {mode['color2']});
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        }}
        .hero h1 {{
            margin: 0;
            font-size: 2rem;
            color: white;
        }}
        .hero p {{
            margin: 0.3rem 0 0 0;
            font-size: 1.05rem;
            opacity: 0.92;
        }}
        [data-testid="stChatMessage"] {{
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            background: rgba(128, 128, 128, 0.10);
            border: 1px solid rgba(128, 128, 128, 0.18);
        }}
        [data-testid="stChatInput"] textarea:focus {{
            border-color: {mode['color1']} !important;
            box-shadow: 0 0 0 1px {mode['color1']} !important;
        }}
        [data-testid="stSidebar"] {{
            border-right: 3px solid {mode['color1']};
        }}
        #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <h1>{mode['icon']} {mode_name} Mode</h1>
        <p>{mode['tagline']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Conversation state (reset when the mode changes) ----------
if st.session_state.get("mode_name") != mode_name:
    st.session_state.mode_name = mode_name
    st.session_state.messages = [SystemMessage(content=mode["prompt"])]

# ---------- Show history ----------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar=mode["icon"]):
            st.write(msg.content)

# ---------- Chat input ----------
prompt = st.chat_input(f"Talk to the {mode_name.lower()} bot...")

if prompt and prompt.strip():
    prompt = prompt.strip()

    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    response = model.invoke(st.session_state.messages)

    with st.chat_message("assistant", avatar=mode["icon"]):
        st.write(response.content)
    st.session_state.messages.append(AIMessage(content=response.content))
