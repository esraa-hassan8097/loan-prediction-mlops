from langgraph.types import interrupt

from src.agent.rag_tool import search_loan_policy
from src.agent.shap_tool import explain_loan
from src.agent.state import LoanAgentState
from src.agent.tools import predict_loan


def prediction_node(state: LoanAgentState):
    """
    Run the existing ML loan prediction
    using applicant information from the state.

    IMPORTANT:
    This node must only be reached after the
    applicant has provided the required signature.
    """
    applicant = state.get("applicant")

    if not applicant:
        raise ValueError(
            "Applicant data is required for loan prediction."
        )

    signature_status = state.get("signature_status")

    if signature_status != "signed":
        raise ValueError(
            "Prediction cannot run before "
            "the applicant signs the application."
        )

    print("\n[PREDICTION NODE]")
    print("Signature verified.")
    print("Calling predict_loan()...")

    prediction = predict_loan(**applicant)

    print(
        f"[PREDICTION NODE] Prediction result: {prediction}"
    )

    return {
        "prediction": prediction
    }


def human_signature_node(state: LoanAgentState):
    """
    Pause the LangGraph workflow and wait for the
    applicant to sign the loan application.

    The prediction has NOT been made yet.

    interrupt() pauses the current graph execution.
    The value passed to interrupt() is sent to the
    caller, for example the Flask GUI.

    When the graph is resumed with Command(resume=...),
    interrupt() returns the supplied resume value.
    """
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

    human_signature = interrupt(signature_request)

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


def explanation_node(state: LoanAgentState):
    """
    Generate a SHAP explanation for the applicant.
    """
    applicant = state.get("applicant")

    if not applicant:
        raise ValueError(
            "Applicant data is required for explanation."
        )

    print("\n[EXPLANATION NODE]")
    print("Calling explain_loan()...")

    explanation = explain_loan(**applicant)

    return {
        "explanation": explanation
    }


def policy_node(state: LoanAgentState):
    """
    Search the loan policy knowledge base
    using the user's question.
    """
    question = state.get(
        "user_message",
        "",
    )

    if not question:
        raise ValueError(
            "User message is required for policy search."
        )

    print("\n[POLICY NODE]")
    print("Calling search_loan_policy()...")

    policy_context = search_loan_policy(
        question=question
    )

    return {
        "policy_context": policy_context
    }


def router_node(state: LoanAgentState):
    """
    Router node.

    Determines which part of the system should
    handle the user's request and stores that
    decision in the graph state.
    """
    user_message = state.get(
        "user_message",
        "",
    ).lower()

    if "predict" in user_message:
        route = "prediction"

    elif "explain" in user_message:
        route = "explanation"

    elif (
        "policy" in user_message
        or "eligibility" in user_message
        or "credit history" in user_message
    ):
        route = "policy"

    else:
        route = "policy"

    return {
        "route": route
    }


def route_request(state: LoanAgentState):
    """
    Read the routing decision from the state.

    This function is used by LangGraph's
    conditional edges.
    """
    return state["route"]