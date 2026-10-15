from src.rag.retriever import retrieve_policy
from src.genai.llm_client import LLMClient


def answer_question(question: str) -> str:
    """
    Retrieve relevant policy information and use the LLM
    to generate an answer based on that context.
    """

    # 1. Retrieve relevant policy information
    retrieved_documents = retrieve_policy(question, top_k=3)

    # 2. Build the context
    context_parts = []

    for document in retrieved_documents:
        context_parts.append(
            f"Source: {document['source']}\n"
            f"Content: {document['content']}"
        )

    context = "\n\n".join(context_parts)

    # 3. Create the prompt for the LLM
    prompt = f"""
You are an AI assistant for a loan prediction system.

Answer the user's question using ONLY the policy information
provided in the context below.

Do not invent policies or financial rules.

If the context does not contain enough information to answer
the question, clearly say that the information is unavailable.

Important:
The policy provides context only.
The machine learning model remains responsible for the
actual loan prediction.

Context:
{context}

User question:
{question}

Provide a clear and concise answer.
"""

    # 4. Send the context + question to GPT-5-mini
    llm = LLMClient()

    answer = llm.generate(prompt)

    return answer


if __name__ == "__main__":

    question = "How does credit history affect loan assessment?"

    answer = answer_question(question)

    print("\n==============================")
    print("RAG ANSWER")
    print("==============================")

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)