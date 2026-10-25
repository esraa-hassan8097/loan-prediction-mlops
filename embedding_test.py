from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project = AIProjectClient(
    endpoint="https://loan-prediction-mlops-recoucelab.services.ai.azure.com",
    credential=DefaultAzureCredential(),
)

client = project.get_openai_client()


text = "Credit history is an important factor in loan assessment."


response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[text],
)


embedding = response.data[0].embedding


print("Embedding created successfully!")
print("Number of dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])
