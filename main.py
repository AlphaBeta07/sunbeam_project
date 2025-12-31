import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sunbeam Chatbot", layout="centered")

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=180)
st.title("Sunbeam Chatbot")

# ---------------- VECTOR DB ----------------
@st.cache_resource
def load_vectordb():
    loader = DirectoryLoader(
        path="data",
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    def is_meaningful_document(text: str) -> bool:
        text = text.strip()
        if len(text) < 200:
            return False
        if text.isupper():
            return False
        if text.count(" ") < 40:
            return False
        return True

    docs = [d for d in documents if is_meaningful_document(d.page_content)]

    embed_model = OpenAIEmbeddings(
        model="text-embedding-nomic-embed-text-v1.5",
        base_url="http://127.0.0.1:1234/v1",
        api_key="dummy",
        check_embedding_ctx_length=False
    )

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_no_chunking"
    )
    return vectordb

vectordb = load_vectordb()

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = init_chat_model(
    model="google/gemma-3n-e4b",
    model_provider="openai",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy"
)

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- QUICK QUESTIONS ----------------
st.subheader("Quick questions")

col1, col2, col3 = st.columns(3)
clicked_pill = None

if col1.button("What courses does Sunbeam offer?"):
    clicked_pill = "What courses does Sunbeam offer?"

if col2.button("Tell me about internships"):
    clicked_pill = "Tell me about internships"

if col3.button("Where is Sunbeam located?"):
    clicked_pill = "Where is Sunbeam located?"

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- TYPEWRITER EFFECT ----------------
def typewriter_effect(text, speed=0.02):
    placeholder = st.empty()
    current = ""
    for word in text.split():
        current += word + " "
        placeholder.write(current)
        time.sleep(speed)

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask a question about Sunbeam Institute")

if clicked_pill:
    user_input = clicked_pill

# ---------------- CHAT FLOW ----------------
if user_input:
    user_input = user_input.replace("intership", "internship")

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            docs = retriever.invoke(user_input)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
                You are a chatbot that answers questions using Sunbeam Institute website data.

                Rules:
                - Answer strictly from the given context.
                - If information is missing, say:
                "Sorry, This information is not available."
                - Do not add external knowledge.

                Context:
                {context}

                Question:
                {user_input}

                Answer:
                """

            response = llm.invoke(prompt)
            answer = response.content

        typewriter_effect(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
