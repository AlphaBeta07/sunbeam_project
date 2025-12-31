import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
import time

st.set_page_config(page_title="Sunbeam Chatbot", layout="centered")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=180)

st.title("Sunbeam Chatbot")

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

if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Try asking:")
pill_cols = st.columns(4)

pills = [
    "What courses does Sunbeam offer?",
    "Tell me about internships",
    "Locat"
]

clicked_pill = None
for col, pill in zip(pill_cols, pills):
    if col.button(pill):
        clicked_pill = pill

def typewriter_effect(text, speed=0.03):
    placeholder = st.empty()
    displayed_text = ""
    for word in text.split(" "):
        displayed_text += word + " "
        placeholder.markdown(displayed_text)
        time.sleep(speed)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question about Sunbeam Institute")

if clicked_pill:
    user_input = clicked_pill

if user_input:
    user_input = user_input.replace("intership", "internship")

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        retrieved_docs = retriever.invoke(user_input)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt = f"""
        You are a chatbot that answers questions using Sunbeam Institute website data.

        Rules:
        - Answer strictly from the given context.
        - If relevant information is available, summarize it clearly.
        - If information is not available, say:
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

        typewriter_effect(answer, speed=0.025)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
