from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

from src.rag.document_loader import load_documents, chunk_text


# -----------------------------
# 1. Connect to Microsoft Foundry
# -----------------------------

project = AIProjectClient(
    endpoint="https://loan-prediction-mlops-recoucelab.services.ai.azure.com",
    credential=DefaultAzureCredential(),
)

openai_client = project.get_openai_client()


# -----------------------------
# 2. Connect to Azure AI Search
# -----------------------------

search_client = SearchClient(
    endpoint="https://loan-mlops-search.search.windows.net",
    index_name="loan-policy-index",
    credential=DefaultAzureCredential(),
)


# -----------------------------
# 3. Load our policy document
# -----------------------------

documents = load_documents(
    "knowledge/loan_policy"
)


# -----------------------------
# 4. Create chunks
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
# 5. Create embeddings
# -----------------------------

texts = [
    chunk["text"]
    for chunk in all_chunks
]

response = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)


# -----------------------------
# 6. Prepare Azure Search records
# -----------------------------

documents_to_upload = []

for i, (chunk, embedding_data) in enumerate(
    zip(all_chunks, response.data)
):

    documents_to_upload.append({

        "id": str(i + 1),

        "source": chunk["source"],

        "content": chunk["text"],

        "content_vector": embedding_data.embedding

    })


# -----------------------------
# 7. Upload to Azure AI Search
# -----------------------------

result = search_client.upload_documents(
    documents=documents_to_upload
)


# -----------------------------
# 8. Display result
# -----------------------------

print("\n==============================")
print("AZURE AI SEARCH UPLOAD")
print("==============================")

print(
    "Uploaded documents:",
    len(documents_to_upload)
)

print("\nUpload result:")

for item in result:

    print(
        item.key,
        item.succeeded
    )