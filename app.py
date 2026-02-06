import streamlit as st
import json
import os
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:1b")

if not os.path.exists("data"):
    os.makedirs("data")

DATA_FILE = "data/user_profiles.json"


def save_user(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


def load_user():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return None


user = load_user()

if not user:
    st.header("Welcome! Let's create your profile.")
    with st.form("survey"):
        name = st.text_input("Full name")
        department = st.selectbox("Department", ("Engineering", "HR", "Sales", "Other"))
        role = st.text_input(
            "Job title (e.g. Junior Software Engineer, HR Manager, Sales Representative)"
        )

        if st.form_submit_button("Submit"):
            user_data = {"name": name, "department": department, "role": role}
            save_user(user_data)
            st.success("Profile created successfully!")
            st.rerun()
else:
    st.title(f"Welcome back, {user['name']}!")
    st.write(f"I am your {user['department']} onboarding assistant.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about your onboarding..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            contextual_prompt = f"""
            You are a helpful onboarding assistant.
            The user's name is {user['name']}.
            Their department is {user['department']}.
            Their role is {user['role']}.
            Answer the following question: {prompt}
            """

        response = llm.invoke(contextual_prompt)

        st.markdown(response.content)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )

    if st.button("Reset Profile (Start Over)"):
        os.remove(DATA_FILE)
        st.rerun()
