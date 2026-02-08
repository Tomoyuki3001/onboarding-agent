import streamlit as st
import json
import os
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOllama(model="llama3.2:1b")

if not os.path.exists("data"):
    os.makedirs("data")

DATA_FILE = "data/user_profiles.json"
CHAT_FILE = "data/chat_history.json"


def load_chat():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r") as f:
            return json.load(f)
    return []


def save_chat(messages):
    with open(CHAT_FILE, "w") as f:
        json.dump(messages, f)


def save_user(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


def load_user():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return None


def get_vector_store():
    docs_folder = "docs"
    all_chunks = []

    if os.path.exists(docs_folder):
        for filename in os.listdir(docs_folder):
            loader = PyPDFLoader(os.path.join(docs_folder, filename))
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )

            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)

    if not all_chunks:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(
        documents=all_chunks, embedding=embeddings, persist_directory="./chroma_db"
    )
    return vector_store


vector_db = get_vector_store()


def get_company_onboarding_document():
    document_chunks = []
    docs_folder = "docs"

    if os.path.exists(docs_folder):
        for filename in os.listdir(docs_folder):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(docs_folder, filename))
                pages = loader.load()
                for page in pages:
                    document_chunks.append(
                        f"--- SOURCE: {filename} ---\n{page.page_content}\n\n"
                    )
    if not document_chunks:
        st.error(
            "No company onboarding document found. Please upload a PDF file to the 'docs' folder."
        )
    else:
        st.sidebar.markdown(
            f"### Company Onboarding Document ({len(document_chunks)} characters)"
        )

    return "\n".join(document_chunks)


company_onboarding_document = get_company_onboarding_document()

user = load_user()

with st.sidebar:
    st.title("User Profile")
    st.image("https://ui-avatars.com/api/?name=" + user["name"])
    st.write(f"Name: {user['name']}")
    st.write(f"Department: {user['department']}")
    st.write(f"Role: {user['role']}")

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat([])
        st.rerun()


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
        st.session_state.messages = load_chat()

    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about your onboarding..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        relevant_docs = vector_db.similarity_search(prompt, k=3)
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        contextual_prompt = f"""
            You are the {user['department']} onboarding assistant.
            User: {user['name']} ({user['role']} in {user['department']})

            Use the following context to answer the user.
            Context: {context_text}
            Question: {prompt}
            """

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

        for chunk in llm.stream(contextual_prompt):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        save_chat(st.session_state.messages)

    if st.button("Reset Profile (Start Over)"):
        os.remove(DATA_FILE)
        st.rerun()
