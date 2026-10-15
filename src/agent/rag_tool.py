from src.rag.retriever import retrieve_policy


def search_loan_policy(question: str, top_k: int = 3):
    """
    Search the loan policy knowledge base.

    Returns the most relevant policy documents
    retrieved from Azure AI Search.
    """

    results = retrieve_policy(
        question=question,
        top_k=top_k
    )

    return results


if __name__ == "__main__":

    question = "How does credit history affect loan assessment?"

    results = search_loan_policy(question)

    print("\nRAG TOOL RESULTS\n")

    for result in results:

        print(f"ID: {result['id']}")
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']}")
        print(f"Content: {result['content']}")
        print("-" * 60)