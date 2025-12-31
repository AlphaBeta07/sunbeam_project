import os
import time
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sunbeam Chatbot", layout="centered")

st.markdown("""
<style>
/* User chat bubble */
[data-testid="chat-message-user"] {
    border-radius: 22px !important;
    padding: 12px 16px !important;
}

/* Inner text container (important for rounding effect) */
[data-testid="chat-message-user"] > div {
    border-radius: 22px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image("logo.png", width=180)

st.title("Sunbeam Chatbot")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Chat History")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

if st.sidebar.button("New Chat"):
    if st.session_state.current_chat:
        st.session_state.chat_sessions.append(st.session_state.current_chat)
    st.session_state.current_chat = []

for i, chat in enumerate(st.session_state.chat_sessions):
    title = chat[0]["content"] if chat else f"Chat {i+1}"
    if st.sidebar.button(title[:30], key=f"chat_{i}"):
        st.session_state.current_chat = chat

# ---------------- VECTOR DB ----------------
@st.cache_resource
def load_vectordb():
    loader = DirectoryLoader(
        path="data",
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()

    def is_meaningful(text):
        text = text.strip()
        return len(text) > 200 and not text.isupper() and text.count(" ") > 40

    docs = [d for d in documents if is_meaningful(d.page_content)]

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

# ---------------- LLM ----------------
llm = init_chat_model(
    model="google/gemma-3n-e4b",
    model_provider="openai",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy"
)

# ---------------- PILLS ----------------
st.subheader("Try asking:")
pill_cols = st.columns(4)

pills = [
    "What courses does Sunbeam offer?",
    "Tell me about internships",
    "Location"
]

clicked_pill = None
for col, pill in zip(pill_cols, pills):
    if col.button(pill):
        clicked_pill = pill

# ---------------- TYPEWRITER EFFECT ----------------
def typewriter_effect(text, speed=0.025):
    placeholder = st.empty()
    out = ""
    for word in text.split():
        out += word + " "
        placeholder.markdown(out)
        time.sleep(speed)

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.current_chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask anything...")

if clicked_pill:
    user_input = clicked_pill

# ---------------- CHAT LOGIC ----------------
if user_input:
    user_input = user_input.replace("intership", "internship")

    st.session_state.current_chat.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        retrieved_docs = retriever.invoke(user_input)

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        use_rag = bool(context and len(context.strip()) > 300)

        if use_rag:
            prompt = f"""
You are a chatbot that answers using Sunbeam Institute website data only.

Rules:
- Answer strictly from the given context
- Be clear and concise
- Do not use external knowledge

Context:
{context}

Question:
{user_input}

Answer:
"""
        else:
            prompt = f"""
You are a helpful general-purpose chatbot.

Rules:
- Answer normally
- Be polite and clear

Question:
{user_input}

Answer:
"""

        response = llm.invoke(prompt)
        answer = response.content

        typewriter_effect(answer)

    st.session_state.current_chat.append(
        {"role": "assistant", "content": answer}
    )
