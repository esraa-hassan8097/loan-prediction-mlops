import json

from src.genai.llm_client import LLMClient
from src.agent.tools import predict_loan
from src.agent.shap_tool import explain_loan
from src.agent.rag_tool import search_loan_policy
from src.monitoring.tracing import configure_monitoring, tracer


class LoanAgent:

    def __init__(self):
        self.llm = LLMClient()

    def _build_tools(self):

        return [
            {
                "type": "function",
                "name": "predict_loan",
                "description": (
                    "Predict whether a loan application is approved or rejected. "
                    "Use this when the user asks for an actual loan prediction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Gender": {"type": "string"},
                        "Married": {"type": "string"},
                        "Dependents": {"type": "string"},
                        "Education": {"type": "string"},
                        "Self_Employed": {"type": "string"},
                        "ApplicantIncome": {"type": "number"},
                        "CoapplicantIncome": {"type": "number"},
                        "LoanAmount": {"type": "number"},
                        "Loan_Amount_Term": {"type": "number"},
                        "Credit_History": {"type": "number"},
                        "Property_Area": {"type": "string"},
                    },
                    "required": [
                        "Gender",
                        "Married",
                        "Dependents",
                        "Education",
                        "Self_Employed",
                        "ApplicantIncome",
                        "CoapplicantIncome",
                        "LoanAmount",
                        "Loan_Amount_Term",
                        "Credit_History",
                        "Property_Area",
                    ],
                },
            },
            {
                "type": "function",
                "name": "explain_loan",
                "description": (
                    "Explain an individual loan prediction using SHAP. "
                    "Use this when the user asks why a prediction was made."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Gender": {"type": "string"},
                        "Married": {"type": "string"},
                        "Dependents": {"type": "string"},
                        "Education": {"type": "string"},
                        "Self_Employed": {"type": "string"},
                        "ApplicantIncome": {"type": "number"},
                        "CoapplicantIncome": {"type": "number"},
                        "LoanAmount": {"type": "number"},
                        "Loan_Amount_Term": {"type": "number"},
                        "Credit_History": {"type": "number"},
                        "Property_Area": {"type": "string"},
                    },
                    "required": [
                        "Gender",
                        "Married",
                        "Dependents",
                        "Education",
                        "Self_Employed",
                        "ApplicantIncome",
                        "CoapplicantIncome",
                        "LoanAmount",
                        "Loan_Amount_Term",
                        "Credit_History",
                        "Property_Area",
                    ],
                },
            },
            {
                "type": "function",
                "name": "search_loan_policy",
                "description": (
                    "Search the loan policy knowledge base using Azure AI Search. "
                    "Use this when the user asks about loan policy or eligibility rules."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The policy question to search for.",
                        }
                    },
                    "required": ["question"],
                },
            },
        ]

    def _execute_tool(self, tool_name, arguments):

        if tool_name == "predict_loan":
            return predict_loan(**arguments)

        if tool_name == "explain_loan":
            return explain_loan(**arguments)

        if tool_name == "search_loan_policy":
            return search_loan_policy(**arguments)

        raise ValueError(f"Unknown tool: {tool_name}")

    def run(self, user_message, applicant=None):

        # Start one OpenTelemetry span for the complete agent execution.
        #
        # We deliberately do NOT put applicant data or the full user
        # message into telemetry because applicant information may contain
        # sensitive financial/personal information.
        with tracer.start_as_current_span("loan_agent_run") as span:

            span.set_attribute(
                "agent.type",
                "loan_prediction_agent"
            )

            span.set_attribute(
                "agent.user_message_length",
                len(user_message)
            )

            applicant_context = (
                json.dumps(applicant)
                if applicant
                else "No applicant data provided."
            )

            instructions = f"""
You are the LoanPredict AI agent.

You orchestrate three tools:

1. Prediction Tool
   - Produces the actual machine-learning loan prediction.
   - Y means approved.
   - N means rejected.

2. SHAP Tool
   - Explains the machine-learning model's output.
   - It does not make the prediction.

3. RAG Policy Tool
   - Retrieves relevant information from the loan policy knowledge base.

Important rules:

- Never invent a loan prediction.
- Never override the prediction returned by the ML model.
- Use the Prediction Tool when an actual prediction is requested.
- Use the SHAP Tool when an explanation of a prediction is requested.
- Use the RAG Tool when policy information is requested.
- You may use multiple tools when the user's request requires them.
- The policy knowledge base provides context and must not override the ML prediction.

Applicant data:

{applicant_context}
"""

            input_items = [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]

            while True:

                response = self.llm.client.responses.create(
                    model=self.llm.model_name,
                    instructions=instructions,
                    input=input_items,
                    tools=self._build_tools(),
                )

                print(f"model response: {response}")

                tool_calls = [
                    item
                    for item in response.output
                    if item.type == "function_call"
                ]

                if not tool_calls:
                    span.set_attribute(
                        "agent.status",
                        "completed"
                    )

                    return response.output_text

                input_items += response.output

                for tool_call in tool_calls:

                    arguments = json.loads(tool_call.arguments)

                    print(
                        f"\n[AGENT] Calling tool: {tool_call.name}"
                    )

                    print(
                        f"[AGENT] Arguments: {arguments}"
                    )

                    # Create a child span for each agent tool execution.
                    with tracer.start_as_current_span(
                        f"agent_tool.{tool_call.name}"
                    ) as tool_span:

                        tool_span.set_attribute(
                            "agent.tool.name",
                            tool_call.name
                        )

                        result = self._execute_tool(
                            tool_call.name,
                            arguments,
                        )

                        tool_span.set_attribute(
                            "agent.tool.status",
                            "completed"
                        )

                    if isinstance(result, (dict, list)):
                        result = json.dumps(
                            result,
                            default=str
                        )
                    else:
                        result = str(result)

                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": result,
                        }
                    )


if __name__ == "__main__":

    # Configure Azure Monitor + OpenTelemetry
    configure_monitoring()

    agent = LoanAgent()

    answer = agent.run(
        "How does credit history affect loan assessment?"
    )

    print("\nFINAL ANSWER\n")
    print(answer)
