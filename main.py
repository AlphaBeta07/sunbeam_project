import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model

st.set_page_config(page_title="Sunbeam Chatbot", layout="centered")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=160)

st.markdown("### Sunbeam Chatbot")

@st.cache_resource
def load_vectordb():
    loader = DirectoryLoader("data", "**/*.txt", TextLoader)
    docs = loader.load()

    docs = [
        d for d in docs
        if len(d.page_content.strip()) > 200
        and not d.page_content.isupper()
        and d.page_content.count(" ") > 40
    ]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-nomic-embed-text-v1.5",
        base_url="http://127.0.0.1:1234/v1",
        api_key="dummy",
        check_embedding_ctx_length=False
    )

    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="chroma_db"
    )

vectordb = load_vectordb()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

llm = init_chat_model(
    model="google/gemma-3n-e4b",
    model_provider="openai",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy"
)

if "chat" not in st.session_state:
    st.session_state.chat = []

def typewriter(text, speed=0.02):
    placeholder = st.empty()
    current = ""
    for word in text.split():
        current += word + " "
        placeholder.markdown(current)
        time.sleep(speed)


st.subheader("Quick questions")

pill = None
c1, c2, c3 = st.columns(3)

if c1.button("What courses does Sunbeam offer?"):
    pill = "What courses does Sunbeam offer?"

if c2.button("Tell me about internships"):
    pill = "Tell me about internships"

if c3.button("Where is Sunbeam located?"):
    pill = "Where is Sunbeam located?"

for msg in st.session_state.messages:
    with st.chat_message(
        msg["role"],
        avatar="🧑" if msg["role"] == "user" else "🤖"
    ):
        st.markdown(msg["content"])

user_input = st.chat_input("Type a message")

if pill:
    user_input = pill

# ---------------- CHAT FLOW ----------------
if user_input:
    user_input = user_input.replace("intership", "internship")

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
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

        typewriter(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
