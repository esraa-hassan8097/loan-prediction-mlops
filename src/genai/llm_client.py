
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


class LLMClient:
    def __init__(self):
        self.project = AIProjectClient(
            endpoint="https://loan-prediction-mlops-recoucelab.services.ai.azure.com/api/projects/loan-prediction-mlops-lab2",
            credential=DefaultAzureCredential(),
        )

        self.client = self.project.get_openai_client()

        self.model_name = "gpt-5-mini-1"

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )

        return response.output_text


if __name__ == "__main__":
    llm = LLMClient()

    answer = llm.generate(
        "Explain machine learning in two simple sentences."
    )

    print(answer)