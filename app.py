from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


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
        ApplicantIncome=float(request.form["ApplicantIncome"]),
        CoapplicantIncome=float(request.form["CoapplicantIncome"]),
        LoanAmount=float(request.form["LoanAmount"]),
        Loan_Amount_Term=float(request.form["Loan_Amount_Term"]),
        Credit_History=float(request.form["Credit_History"]),
        Property_Area=request.form["Property_Area"]
    )

    features = data.get_data_as_dataframe()

    pipeline = PredictPipeline()

    prediction = pipeline.predict(features)

    return render_template(
        "index.html",
        prediction=prediction[0]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)