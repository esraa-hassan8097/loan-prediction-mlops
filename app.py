import atexit
import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.types import Command

# LangGraph
from src.agent.graph import build_loan_graph
from src.agent.shap_tool import explain_loan
from src.components.production_logger import ProductionLogger
from src.pipeline.prediction_pipeline import (
    CustomData,
    PredictPipeline,
)
from src.rag.rag_pipeline import answer_question
from src.rag.retriever import retrieve_policy

# Load environment variables for LangSmith, Langfuse, Azure, etc.
load_dotenv(dotenv_path=".env")

app = Flask(__name__)





# ============================================================

# LANGGRAPH

# ============================================================



# Build the graph ONCE when Flask starts.

# Do NOT build a new graph inside every request.

# The graph already contains the MemorySaver checkpointer.



loan_graph = build_loan_graph()

# Langfuse production tracing. LangSmith remains enabled independently
# through the existing LANGSMITH_* environment variables.
langfuse_client = get_client()
langfuse_handler = CallbackHandler()

# Keep request telemetry asynchronous; flush pending telemetry on a normal
# process shutdown/restart.
atexit.register(langfuse_client.shutdown)



print("LangGraph loaded successfully.")





# ============================================================

# LANGFUSE PRODUCTION TRACE HELPERS

# ============================================================


def langfuse_common_tags():
    return [
        "loan-assistant",
        "langgraph",
        "flask",
    ]


def langfuse_environment():
    return os.getenv(
        "LANGFUSE_TRACING_ENVIRONMENT",
        "development",
    )


# ============================================================

# HOME / PREDICTION

# ============================================================



@app.route("/")

def home():

    return render_template("index.html")





@app.route("/predict", methods=["POST"])

def predict():



    data = CustomData(

        Gender=request.form["Gender"],

        Married=request.form["Married"],

        Dependents=request.form["Dependents"],

        Education=request.form["Education"],

        Self_Employed=request.form["Self_Employed"],

        ApplicantIncome=float(

            request.form["ApplicantIncome"]

        ),

        CoapplicantIncome=float(

            request.form["CoapplicantIncome"]

        ),

        LoanAmount=float(

            request.form["LoanAmount"]

        ),

        Loan_Amount_Term=float(

            request.form["Loan_Amount_Term"]

        ),

        Credit_History=float(

            request.form["Credit_History"]

        ),

        Property_Area=request.form["Property_Area"],

    )



    features = data.get_data_as_dataframe()



    pipeline = PredictPipeline()



    prediction = pipeline.predict(

        features

    )



    logger = ProductionLogger()



    logger.log_prediction_request(

        input_data=features.iloc[0].to_dict(),

        prediction=prediction[0],

    )



    return render_template(

        "index.html",

        prediction=prediction[0],

    )





# ============================================================

# MONITORING DASHBOARD

# ============================================================



@app.route("/dashboard")

def dashboard():



    try:

        drift_report = pd.read_csv(

            "artifacts/drift_report.csv"

        ).to_dict(

            orient="records"

        )

    except Exception:  # noqa: BLE001

        drift_report = []



    try:

        prediction_report = pd.read_csv(

            "artifacts/prediction_report.csv"

        ).to_dict(

            orient="records"

        )

    except Exception:  # noqa: BLE001

        prediction_report = []



    try:

        performance_report = pd.read_csv(

            "artifacts/performance_report.csv"

        ).to_dict(

            orient="records"

        )

    except Exception:  # noqa: BLE001

        performance_report = []



    return render_template(

        "dashboard.html",

        drift_report=drift_report,

        prediction_report=prediction_report,

        performance_report=performance_report,

    )





# ============================================================

# SHAP EXPLAINABILITY

# ============================================================



@app.route(

    "/shap",

    methods=["GET", "POST"]

)

def shap_page():



    contributions = []

    prediction = None



    if request.method == "GET":

        return render_template(

            "shap.html",

            contributions=[],

            prediction=None,

        )



    data = CustomData(

        Gender=request.form["Gender"],

        Married=request.form["Married"],

        Dependents=request.form["Dependents"],

        Education=request.form["Education"],

        Self_Employed=request.form["Self_Employed"],

        ApplicantIncome=float(

            request.form["ApplicantIncome"]

        ),

        CoapplicantIncome=float(

            request.form["CoapplicantIncome"]

        ),

        LoanAmount=float(

            request.form["LoanAmount"]

        ),

        Loan_Amount_Term=float(

            request.form["Loan_Amount_Term"]

        ),

        Credit_History=float(

            request.form["Credit_History"]

        ),

        Property_Area=request.form["Property_Area"],

    )



    features = data.get_data_as_dataframe()



    pipeline = PredictPipeline()



    prediction = pipeline.predict(

        features

    )[0]



    contributions = explain_loan(

        Gender=data.Gender,

        Married=data.Married,

        Dependents=data.Dependents,

        Education=data.Education,

        Self_Employed=data.Self_Employed,

        ApplicantIncome=data.ApplicantIncome,

        CoapplicantIncome=data.CoapplicantIncome,

        LoanAmount=data.LoanAmount,

        Loan_Amount_Term=data.Loan_Amount_Term,

        Credit_History=data.Credit_History,

        Property_Area=data.Property_Area,

    )



    return render_template(

        "shap.html",

        contributions=contributions[:8],

        prediction=prediction,

    )





# ============================================================

# RAG POLICY ASSISTANT

# ============================================================



@app.route(

    "/rag",

    methods=["GET", "POST"]

)

def rag_page():



    question = ""

    answer = None

    documents = []



    if request.method == "POST":



        question = request.form[

            "question"

        ]



        documents = retrieve_policy(

            question=question,

            top_k=3,

        )



        answer = answer_question(

            question

        )



    return render_template(

        "rag.html",

        question=question,

        answer=answer,

        documents=documents,

    )





# ============================================================

# LANGGRAPH ASSISTANT

# ============================================================



@app.route(

    "/assistant",

    methods=["GET", "POST"]

)

def assistant_page():



    question = ""

    applicant = None

    result = None

    error = None

    thread_id = None



    # --------------------------------------------------------

    # GET

    # --------------------------------------------------------



    if request.method == "GET":



        return render_template(

            "assistant.html",

            question="",

            applicant=None,

            result=None,

            error=None,

            thread_id=None,

        )



    # --------------------------------------------------------

    # Get user question

    # --------------------------------------------------------



    question = request.form.get(

        "question",

        ""

    ).strip()



    if not question:



        error = "Please enter a question."



        return render_template(

            "assistant.html",

            question=question,

            applicant=None,

            result=None,

            error=error,

            thread_id=None,

        )



    # --------------------------------------------------------

    # Applicant fields

    # --------------------------------------------------------



    applicant_fields = [

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

    ]



    supplied_fields = [

        field

        for field in applicant_fields

        if request.form.get(

            field,

            ""

        ).strip()

    ]



    # --------------------------------------------------------

    # Applicant is optional for policy questions.

    # If any applicant fields are supplied,

    # all applicant fields must be supplied.

    # --------------------------------------------------------



    if supplied_fields:



        missing_fields = [

            field

            for field in applicant_fields

            if not request.form.get(

                field,

                ""

            ).strip()

        ]



        if missing_fields:



            error = (

                "Please complete all applicant fields: "

                + ", ".join(

                    missing_fields

                )

            )



            return render_template(

                "assistant.html",

                question=question,

                applicant=None,

                result=None,

                error=error,

                thread_id=None,

            )



        # ----------------------------------------------------

        # Convert applicant data

        # ----------------------------------------------------



        try:



            applicant = {

                "Gender":

                    request.form["Gender"],



                "Married":

                    request.form["Married"],



                "Dependents":

                    request.form["Dependents"],



                "Education":

                    request.form["Education"],



                "Self_Employed":

                    request.form["Self_Employed"],



                "ApplicantIncome":

                    float(

                        request.form[

                            "ApplicantIncome"

                        ]

                    ),



                "CoapplicantIncome":

                    float(

                        request.form[

                            "CoapplicantIncome"

                        ]

                    ),



                "LoanAmount":

                    float(

                        request.form[

                            "LoanAmount"

                        ]

                    ),



                "Loan_Amount_Term":

                    float(

                        request.form[

                            "Loan_Amount_Term"

                        ]

                    ),



                "Credit_History":

                    float(

                        request.form[

                            "Credit_History"

                        ]

                    ),



                "Property_Area":

                    request.form[

                        "Property_Area"

                    ],

            }



        except ValueError:



            error = (

                "Please enter valid numbers in the "

                "financial and credit history fields."

            )



            return render_template(

                "assistant.html",

                question=question,

                applicant=None,

                result=None,

                error=error,

                thread_id=None,

            )



    # --------------------------------------------------------

    # Create a NEW LangGraph thread

    # --------------------------------------------------------



    thread_id = (

        "loan-"

        + uuid.uuid4().hex

    )



    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "callbacks": [langfuse_handler],
        "run_name": "loan-agent",
    }



    # --------------------------------------------------------

    # Start LangGraph

    # --------------------------------------------------------



    try:



        # Create a Langfuse root observation around the LangGraph execution.
        # thread_id is reused as session_id so the request and later resume
        # belong to the same Langfuse session.
        with langfuse_client.start_as_current_observation(
            as_type="agent",
            name="loan-assistant-request",
            input={
                "question": question,
                "has_applicant": applicant is not None,
            },
        ) as root_observation:
            with propagate_attributes(
                trace_name="loan-assistant",
                session_id=thread_id,
                tags=langfuse_common_tags(),
                metadata={
                    "channel": "web",
                    "workflow": "loan_assistant",
                },
                environment=langfuse_environment(),
            ):
                result = loan_graph.invoke(
                    {
                        "user_message": question,
                        "applicant": applicant,
                    },
                    config=config,
                )

            route = result.get("route") if isinstance(result, dict) else None
            root_observation.update(
                output={
                    "route": route,
                    "interrupted": (
                        isinstance(result, dict)
                        and "__interrupt__" in result
                    ),
                },
                metadata={
                    "route": route or "unknown",
                    "thread_id": thread_id,
                },
            )

        print(

            "\n=============================="

        )

        print(

            "LANGGRAPH GUI REQUEST"

        )

        print(

            "=============================="

        )



        print(

            "Thread ID:"

        )

        print(

            thread_id

        )



        print(

            "\nGraph Result:"

        )

        print(

            result

        )



    except Exception:  



        app.logger.exception(

            "LangGraph assistant request failed"

        )



        error = (

            "The LangGraph Assistant could not "

            "complete your request."

        )



    return render_template(

        "assistant.html",

        question=question,

        applicant=applicant,

        result=result,

        error=error,

        thread_id=thread_id,

    )





# ============================================================

# LANGGRAPH RESUME

# ============================================================



@app.route(

    "/assistant/resume",

    methods=["POST"]

)

def assistant_resume():



    # --------------------------------------------------------

    # Get thread ID from hidden HTML field

    # --------------------------------------------------------



    thread_id = request.form.get(

        "thread_id"

    )



    if not thread_id:



        return render_template(

            "assistant.html",

            question="",

            applicant=None,

            result=None,

            error=(

                "Missing LangGraph thread ID."

            ),

            thread_id=None,

        )



    # --------------------------------------------------------

    # Get applicant signature

    # --------------------------------------------------------



    signature = request.form.get(

        "signature",

        ""

    ).strip()



    signed_by = request.form.get(

        "signed_by",

        ""

    ).strip()



    # --------------------------------------------------------

    # Validate signature

    # --------------------------------------------------------



    if not signature:



        return render_template(

            "assistant.html",

            question="",

            applicant=None,

            result=None,

            error=(

                "Please provide your signature."

            ),

            thread_id=thread_id,

        )



    if not signed_by:



        return render_template(

            "assistant.html",

            question="",

            applicant=None,

            result=None,

            error=(

                "Please enter the applicant name."

            ),

            thread_id=thread_id,

        )



    # --------------------------------------------------------

    # IMPORTANT:

    # Use the SAME thread_id.


    # LangGraph uses this to locate the paused checkpoint.

    # --------------------------------------------------------



    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "callbacks": [langfuse_handler],
        "run_name": "loan-agent-resume",
    }



    # --------------------------------------------------------

    # Human signature response


    # This dictionary becomes the return value of

    # interrupt() inside human_signature_node().

    # --------------------------------------------------------



    human_response = {

        "signed": True,

        "signature": signature,

        "signed_by": signed_by,

    }



    print(

        "\n=============================="

    )

    print(

        "RESUMING LANGGRAPH"

    )

    print(

        "=============================="

    )



    print(

        "Thread ID:"

    )

    print(

        thread_id

    )



    print(

        "\nHuman response:"

    )

    print(

        human_response

    )



    # --------------------------------------------------------

    # Resume interrupted LangGraph

    # --------------------------------------------------------



    try:



        # Reuse the same Langfuse session_id/thread_id for the resume step.
        with langfuse_client.start_as_current_observation(
            as_type="agent",
            name="loan-assistant-resume",
            input={
                "resume": True,
                "signed": True,
            },
        ) as root_observation:
            with propagate_attributes(
                trace_name="loan-assistant",
                session_id=thread_id,
                tags=langfuse_common_tags(),
                metadata={
                    "channel": "web",
                    "workflow": "loan_assistant",
                    "resume": "true",
                },
                environment=langfuse_environment(),
            ):
                result = loan_graph.invoke(
                    Command(
                        resume=human_response
                    ),
                    config=config,
                )

            route = result.get("route") if isinstance(result, dict) else None
            root_observation.update(
                output={
                    "route": route,
                    "prediction_available": (
                        isinstance(result, dict)
                        and result.get("prediction") is not None
                    ),
                },
                metadata={
                    "route": route or "unknown",
                    "thread_id": thread_id,
                },
            )

        print(

            "\nFinal Graph Result:"

        )

        print(

            result

        )



    except Exception:  



        app.logger.exception(

            "LangGraph resume failed"

        )



        return render_template(

            "assistant.html",

            question="",

            applicant=None,

            result=None,

            error=(

                "The LangGraph workflow could not "

                "be resumed."

            ),

            thread_id=thread_id,

        )



    # --------------------------------------------------------

    # Display final result

    # --------------------------------------------------------



    return render_template(

        "assistant.html",

        question="",

        applicant=None,

        result=result,

        error=None,

        thread_id=thread_id,

    )





# ============================================================

# START FLASK

# ============================================================



if __name__ == "__main__":



    app.run(

        host="0.0.0.0", # nosec B104 - required for container networking

        port=5000,

        debug=False,

    )
