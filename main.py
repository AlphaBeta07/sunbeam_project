import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model



# --------------------------------------------------
# 2. FILTER LOW-QUALITY DOCUMENTS (CRITICAL)
# --------------------------------------------------
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
print(f"Kept {len(docs)} meaningful documents")

# --------------------------------------------------
# 3. EMBEDDINGS
# --------------------------------------------------
embed_model = OpenAIEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy",
    check_embedding_ctx_length=False
)

# --------------------------------------------------
# 4. VECTOR STORE (DOCUMENT-LEVEL)
# --------------------------------------------------
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory="chroma_db_no_chunking"
)

# --------------------------------------------------
# 5. RETRIEVER (SIMPLE SIMILARITY)
# --------------------------------------------------
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# 6. LLM
# --------------------------------------------------
llm = init_chat_model(
    model="google/gemma-3n-e4b",
    model_provider="openai",
    base_url="http://127.0.0.1:1234/v1",
    api_key="dummy"
)

SYSTEM_PROMPT = """
You are a chatbot that answers questions using Sunbeam Institute website data.

Rules:
- Answer strictly from the given context.
- If relevant information is available, summarize it clearly.
- If information is not available, say:
  "This information is not available on the Sunbeam website."
- Do not add external knowledge.
"""

# --------------------------------------------------
# 7. CHAT LOOP
# --------------------------------------------------
while True:
    question = input("\nAsk a question (type 'exit' to quit): ")
    if question.lower() == "exit":
        break

    question = question.replace("intership", "internship")

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)
    print("\nAnswer:\n", response.content)
