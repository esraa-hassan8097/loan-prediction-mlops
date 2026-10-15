from typing import Any, TypedDict


class LoanAgentState(TypedDict, total=False):
    user_message: str
    applicant: dict[str, Any] | None

    messages: list[Any]

    route: str

    prediction: str | None
    explanation: Any | None
    policy_context: Any | None

