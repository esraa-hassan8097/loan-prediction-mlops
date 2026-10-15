from pathlib import Path


def load_documents(folder_path):
    documents = []

    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":

    documents = load_documents(
        "knowledge/loan_policy"
    )

    for document in documents:

        chunks = chunk_text(
            document["text"]
        )

        print("\nSOURCE:", document["source"])
        print("NUMBER OF CHUNKS:", len(chunks))

        for i, chunk in enumerate(chunks):

            print(f"\n--- CHUNK {i + 1} ---")
            print(chunk)
