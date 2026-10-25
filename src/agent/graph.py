import json
from typing import Any, TypedDict

from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agent.rag_tool import search_loan_policy
from src.agent.shap_tool import explain_loan
from src.agent.tools import predict_loan
from src.genai.llm_client import LLMClient


class LoanState(TypedDict, total=False):

    user_message: str

    applicant: dict | None

    route: str

    prediction: Any

    explanation: Any

    policy_context: Any

    out_of_scope_message: Any

    signature_request: Any

    human_signature: Any

    signature_status: str





llm = LLMClient()

# Langfuse callback for LangGraph tracing.
# LangSmith remains enabled independently through LANGSMITH_* environment variables.
langfuse_handler = CallbackHandler()





def llm_router_node(state: LoanState):

    user_message = state.get("user_message", "")

    applicant = state.get("applicant")



    router_prompt = (

        "You are the routing component of a loan AI assistant.\n\n"

        "Your job is ONLY to determine which workflow should handle "

        "the user's request.\n\n"

        "Available routes:\n\n"

        "1. prediction\n"

        "Use when the user wants an actual ML loan prediction.\n\n"

        "2. explanation\n"

        "Use when the user wants an explanation of an existing "

        "loan prediction or asks why the model made a prediction.\n\n"

        "3. policy\n"

        "Use when the user asks about loan policy, eligibility, "

        "rules, requirements, or general loan-policy information.\n\n"

        "4. out_of_scope\n"

        "Use when the user asks about anything unrelated to loan prediction, "

        "loan prediction explanation, or loan policy. Examples include weather, "

        "sports, politics, travel, medical advice, programming, or unrelated "

        "general conversation.\n\n"

        "Return ONLY valid JSON.\n\n"

        'Example: {"route": "prediction"}\n\n'

        "Do not answer the user's question.\n"

        "Do not make a loan prediction.\n"

        "Only select the route.\n\n"

        f"User message:\n{user_message}\n\n"

        f"Applicant data:\n{json.dumps(applicant, default=str)}"

    )



    response = llm.client.responses.create(

        model=llm.model_name,

        instructions=router_prompt,

        input=[

            {

                "role": "user",

                "content": user_message,

            }

        ],

    )



    raw_output = response.output_text.strip()



    print("\n[LLM ROUTER]")

    print("Raw response:")

    print(raw_output)



    try:

        decision = json.loads(raw_output)



    except json.JSONDecodeError as exc:

        raise ValueError(

            f"LLM router did not return valid JSON: {raw_output}"

        ) from exc



    route = decision.get("route")



    allowed_routes = {

        "prediction",

        "explanation",

        "policy",

        "out_of_scope",

    }



    if route not in allowed_routes:

        raise ValueError(

            f"Invalid route returned by LLM: {route}"

        )



    print(

        f"[LLM ROUTER] Selected route: {route}"

    )



    return {

        "route": route

    }





def route_request(state: LoanState):

    return state["route"]





def human_signature_node(state: LoanState):

    applicant = state.get("applicant")



    if not applicant:

        raise ValueError(

            "Applicant data is required before signature."

        )



    signature_request = {

        "type": "loan_application_signature",

        "message": (

            "Please confirm that the information in your "

            "loan application is correct and sign to submit "

            "the application."

        ),

        "status": "awaiting_signature",

    }



    print("\n[HUMAN SIGNATURE]")

    print(

        "The graph is pausing and waiting "

        "for the applicant signature."

    )

    print("Prediction has NOT been executed.")



    human_signature = interrupt(

        signature_request

    )



    print("\n[HUMAN SIGNATURE]")

    print("Graph resumed.")

    print(

        f"Human signature response: {human_signature}"

    )



    if not isinstance(human_signature, dict):

        raise TypeError(

            "Invalid signature response."

        )



    signed = human_signature.get(

        "signed",

        False,

    )



    if not signed:

        print(

            "[HUMAN SIGNATURE] "

            "Signature was not provided."

        )



        return {

            "signature_request": signature_request,

            "human_signature": human_signature,

            "signature_status": "not_signed",

        }



    signature = human_signature.get(

        "signature",

        "",

    )



    signed_by = human_signature.get(

        "signed_by",

        "",

    )



    if not signature or not signed_by:

        raise ValueError(

            "Signature and signer name are required."

        )



    print(

        "[HUMAN SIGNATURE] "

        "Application signed successfully."

    )



    return {

        "signature_request": signature_request,

        "human_signature": human_signature,

        "signature_status": "signed",

    }





def prediction_node(state: LoanState):

    applicant = state.get("applicant")



    if not applicant:

        raise ValueError(

            "Applicant data is required "

            "for loan prediction."

        )



    signature_status = state.get(

        "signature_status"

    )



    if signature_status != "signed":

        raise ValueError(

            "Prediction cannot run before "

            "the applicant signs the application."

        )



    print("\n[PREDICTION NODE]")

    print("Signature verified.")

    print("Calling predict_loan()...")



    result = predict_loan(

        **applicant

    )



    print(

        f"[PREDICTION NODE] Prediction result: {result}"

    )



    return {

        "prediction": result

    }





def explanation_node(state: LoanState):

    applicant = state.get("applicant")



    if not applicant:

        raise ValueError(

            "Applicant data is required "

            "for explanation."

        )



    print("\n[EXPLANATION NODE]")

    print("Calling explain_loan()...")



    result = explain_loan(

        **applicant

    )



    return {

        "explanation": result

    }





def policy_node(state: LoanState):

    question = state.get(

        "user_message",

        "",

    )



    if not question:

        raise ValueError(

            "User message is required "

            "for policy search."

        )



    print("\n[POLICY NODE]")

    print("Calling search_loan_policy()...")



    result = search_loan_policy(

        question=question

    )



    return {

        "policy_context": result

    }





def out_of_scope_node(state: LoanState):
    user_message = state.get("user_message", "")

    print("\n[OUT OF SCOPE NODE]")
    print(f"Unsupported request: {user_message}")

    return {
        "out_of_scope_message": (
            "I can only assist with loan prediction, "
            "loan prediction explanations, and loan policy questions."
        )
    }


def build_loan_graph():

    graph = StateGraph(

        LoanState

    )



    graph.add_node(

        "router",

        llm_router_node





    )



    graph.add_node(

        "human_signature",

        human_signature_node

    )



    graph.add_node(

        "prediction",

        prediction_node

    )



    graph.add_node(

        "explanation",

        explanation_node

    )



    graph.add_node(

        "policy",

        policy_node

    )



    graph.add_node(

        "out_of_scope",

        out_of_scope_node

    )



    graph.add_edge(

        START,

        "router"

    )



    graph.add_conditional_edges(

        "router",

        route_request,

        {

            "prediction": "human_signature",

            "explanation": "explanation",

            "policy": "policy",

            "out_of_scope": "out_of_scope",

        },

    )



    graph.add_edge(

        "human_signature",

        "prediction"

    )



    graph.add_edge(

        "prediction",

        END

    )



    graph.add_edge(

        "explanation",

        END

    )



    graph.add_edge(

        "policy",

        END

    )



    graph.add_edge(

        "out_of_scope",

        END

    )



    memory = MemorySaver()



    return graph.compile(

        checkpointer=memory

    )





if __name__ == "__main__":

    loan_graph = build_loan_graph()



    print("LangGraph compiled successfully.")



    applicant_a = {

        "Gender": "Male",

        "Married": "Yes",

        "Dependents": "0",

        "Education": "Graduate",

        "Self_Employed": "No",

        "ApplicantIncome": 5000,

        "CoapplicantIncome": 2000,

        "LoanAmount": 150,

        "Loan_Amount_Term": 360,

        "Credit_History": 1,

        "Property_Area": "Urban",

    }



    applicant_b = {

        "Gender": "Female",

        "Married": "No",

        "Dependents": "2",

        "Education": "Graduate",

        "Self_Employed": "No",

        "ApplicantIncome": 3000,

        "CoapplicantIncome": 0,

        "LoanAmount": 120,

        "Loan_Amount_Term": 360,

        "Credit_History": 1,

        "Property_Area": "Rural",

    }



    config_a = {

        "configurable": {

            "thread_id": "loan-001"

        }

    }



    config_b = {

        "configurable": {

            "thread_id": "loan-002"

        }

    }



    print("\n===================================")

    print("STARTING THREAD A")

    print("===================================")



    result_a = loan_graph.invoke(

        {

            "user_message": (

                "Predict whether this loan application "

                "will be approved."

            ),

            "applicant": applicant_a,

        },

        config=config_a,

    )



    print("\nTHREAD A RESULT")

    print(result_a)



    print("\n===================================")

    print("STARTING THREAD B")

    print("===================================")



    result_b = loan_graph.invoke(

        {

            "user_message": (

                "Predict whether this loan application "

                "will be approved."

            ),

            "applicant": applicant_b,

        },

        config=config_b,

    )



    print("\nTHREAD B RESULT")

    print(result_b)



    print("\n===================================")

    print("CHECKING BOTH THREADS")

    print("===================================")



    thread_a_paused = "__interrupt__" in result_a

    thread_b_paused = "__interrupt__" in result_b



    print(

        f"Thread A paused: {thread_a_paused}"

    )



    print(

        f"Thread B paused: {thread_b_paused}"

    )



    print("\n===================================")

    print("RESUMING THREAD A ONLY")

    print("===================================")



    final_a = loan_graph.invoke(

        Command(

            resume={

                "signed": True,

                "signature": "signature-A",

                "signed_by": "Applicant A",

            }

        ),

        config=config_a,

    )



    print("\nTHREAD A FINAL RESULT")

    print(final_a)



    print("\n===================================")

    print("CHECKING THREAD B AFTER THREAD A")

    print("===================================")



    state_b = loan_graph.get_state(

        config_b

    )



    print("\nTHREAD B CURRENT STATE")

    print(state_b)



    print("\n===================================")

    print("CHECKPOINT TEST SUMMARY")

    print("===================================")



    print("\nThread A:")

    print("  Thread ID: loan-001")

    print("  Status: RESUMED")

    print("  Signature: provided")

    print(

        f"  Prediction: {final_a.get('prediction')}"

    )



    print("\nThread B:")

    print("  Thread ID: loan-002")

    print("  Status: STILL WAITING")

    print("  Signature: not provided")

    print("  Prediction: NOT executed")



    print(

        "\nCheckpoint isolation test completed."

    )