import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model

# =========================
# 1. LOAD DOCUMENTS
# =========================
loader = DirectoryLoader(
    path=r"D:\Python\IIT-08-H-A-GENERATIVE-AI-94735\Project\data",
    glob="**/*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

# add metadata
for doc in docs:
    doc.metadata["id"] = os.path.splitext(
        os.path.basename(doc.metadata["source"])
    )[0]

print(f"✅ Loaded {len(docs)} documents")

# =========================
# 2. EMBEDDINGS (LOCAL SAFE)
# =========================
embed_model = OpenAIEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy",
    check_embedding_ctx_length=False
)

# =========================
# 3. VECTOR STORE
# =========================
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory="chroma_db_no_split"
)
vectordb.persist()

print("✅ Embeddings stored in ChromaDB")

# =========================
# 4. RETRIEVER
# =========================
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# =========================
# 5. LLM
# =========================
llm = init_chat_model(
    model="google/gemma-3n-e4b",
    model_provider="openai",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy"
)

SYSTEM_PROMPT = """
You are an AI assistant built to answer questions about Sunbeam Institute of Information Technology.

You must strictly follow these rules:

1. Answer ONLY using the information retrieved from the Sunbeam website data provided to you.
2. If the answer is not present in the retrieved context, clearly say:
   "This information is not available on the Sunbeam website."
3. Do NOT guess, assume, or add external knowledge.
4. Be clear, concise, and factual.
5. If a question is ambiguous, ask the user for clarification instead of guessing.
6. When listing information (courses, features, eligibility, duration, fees, etc.), present it in a clean and readable format.
7. Do NOT mention internal system details like embeddings, vector databases, or scraping.
8. Your role is informational, not promotional.

Your knowledge domain includes:
- About Sunbeam Institute
- Internship and Training Programs
- Modular Courses and their details

Always prioritize accuracy over verbosity.

"""

# =========================
# 6. CHAT LOOP (RAG)
# =========================
while True:
    user_input = input("\nEnter your question (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(user_input)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{user_input}

Answer (short, point-wise):
"""

    response = llm.invoke(prompt)
    print("\n🤖 Answer:\n", response.content)
