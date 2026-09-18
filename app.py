import streamlit as st
import json
from query import get_retriever, ask
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Biodiversity Intelligence Chatbot", layout="centered")
st.title("🌱 AI Biodiversity Intelligence Chatbot")
st.caption("Ask about soil, land, biodiversity, and climate conditions to get evidence-backed recommendations.")

# Initialize resources once (cached across reruns)
@st.cache_resource
def load_resources():
    db = get_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=os.getenv("GOOGLE_API_KEY"))
    return db, llm

db, llm = load_resources()

# Session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: input mode toggle
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio("Choose input type:", ["Text", "Structured (JSON)"])

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle input based on mode
user_input = None

if input_mode == "Text":
    user_input = st.chat_input("Describe your land, soil, or biodiversity concern...")

else:
    st.sidebar.markdown("Enter structured data as JSON:")
    json_input = st.sidebar.text_area(
        "JSON input",
        value='{\n  "soil_organic_carbon": 0.3,\n  "rainfall": "low",\n  "crop": "monoculture wheat",\n  "region": "semi-arid"\n}',
        height=180
    )
    if st.sidebar.button("Submit JSON"):
        try:
            data = json.loads(json_input)
            user_input = ", ".join([f"{k.replace('_',' ')}: {v}" for k, v in data.items()])
            user_input += ". What should I do?"
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON — please check the format.")

# Process the input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing environmental data..."):
            answer = ask(user_input, db, llm, st.session_state.history)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history += f"User: {user_input}\nAssistant: {answer}\n"