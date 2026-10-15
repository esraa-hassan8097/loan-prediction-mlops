from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


# Azure AI Foundry
project = AIProjectClient(
    endpoint="https://loan-prediction-mlops-recoucelab.services.ai.azure.com",
    credential=DefaultAzureCredential(),
)

openai_client = project.get_openai_client()


# Azure AI Search
search_client = SearchClient(
    endpoint="https://loan-mlops-search.search.windows.net",
    index_name="loan-policy-index",
    credential=DefaultAzureCredential(),
)


def retrieve_policy(question: str, top_k: int = 3):
    """
    Retrieve the most relevant policy chunks for a user question.
    """

    # 1. Convert the user's question into an embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )

    query_vector = response.data[0].embedding

    # 2. Create a vector search query
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    # 3. Search Azure AI Search
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "source", "content"],
    )

    # 4. Collect the retrieved documents
    retrieved_documents = []

    for result in results:
        retrieved_documents.append(
            {
                "id": result["id"],
                "source": result["source"],
                "content": result["content"],
                "score": result["@search.score"],
            }
        )

    return retrieved_documents


# Test the retriever
if __name__ == "__main__":

    question = "How does credit history affect loan assessment?"

    results = retrieve_policy(question)

    print("\n==============================")
    print("RAG RETRIEVAL TEST")
    print("==============================")

    print("\nQuestion:")
    print(question)

    for i, result in enumerate(results, start=1):

        print(f"\n--- RESULT {i} ---")
        print("ID:", result["id"])
        print("Source:", result["source"])
        print("Score:", result["score"])
        print("Content:")
        print(result["content"])