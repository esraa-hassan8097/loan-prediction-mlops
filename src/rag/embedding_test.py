from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from src.rag.document_loader import chunk_text, load_documents

# -----------------------------
# 1. Connect to Microsoft Foundry
# -----------------------------

project = AIProjectClient(
    endpoint="https://loan-prediction-mlops-recoucelab.services.ai.azure.com",
    credential=DefaultAzureCredential(),
)

client = project.get_openai_client()


# -----------------------------
# 2. Load our policy document
# -----------------------------

documents = load_documents(
    "knowledge/loan_policy"
)


# -----------------------------
# 3. Create chunks
# -----------------------------

all_chunks = []

for document in documents:

    chunks = chunk_text(
        document["text"]
    )

    for chunk in chunks:

        all_chunks.append({
            "source": document["source"],
            "text": chunk
        })


# -----------------------------
# 4. Extract chunk text
# -----------------------------

texts = [
    chunk["text"]
    for chunk in all_chunks
]


# -----------------------------
# 5. Create Azure embeddings
# -----------------------------

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)


# -----------------------------
# 6. Attach embeddings to chunks
# -----------------------------

for chunk, embedding_data in zip(
    all_chunks,
    response.data
):

    chunk["embedding"] = embedding_data.embedding


# -----------------------------
# 7. Display results
# -----------------------------

print("\n==============================")
print("RAG EMBEDDING TEST")
print("==============================")

print("Number of chunks:", len(all_chunks))

for i, chunk in enumerate(all_chunks):

    print(f"\n--- CHUNK {i + 1} ---")

    print("Source:", chunk["source"])

    print(
        "Text:",
        chunk["text"][:300],
        "..."
    )

    print(
        "Embedding dimensions:",
        len(chunk["embedding"])
    )

    print(
        "First 10 values:",
        chunk["embedding"][:10]
    )
