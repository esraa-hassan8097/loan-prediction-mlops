import pickle  # pickle is Python's built-in library for saving and loading Python objects.
import pandas as pd


class PredictPipeline:

    def __init__(self):
        pass

    def load_objects(self):
        # This method loads the preprocessor and the selected trained model
        # from their respective pickle files.

        with open(
            "artifacts/preprocessor.pkl",
            "rb"
        ) as file:

            # Read the bytes inside preprocessor.pkl and rebuild
            # the original preprocessing object.
            preprocessor = pickle.load(file)

        with open(
            "artifacts/model.pkl",
            "rb"
        ) as file:

            # Read the bytes inside model.pkl and rebuild
            # the selected trained model.
            model = pickle.load(file)

        return (
            preprocessor,
            model
        )

    def predict(self, features):
        # features is the DataFrame returned by
        # CustomData.get_data_as_dataframe()

        preprocessor, model = self.load_objects()

        # Apply the exact same preprocessing used during training.
        transformed_features = preprocessor.transform(features)

        # Make the machine-learning prediction.
        prediction = model.predict(transformed_features)

        return prediction


class CustomData:

    def __init__(
        self,
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area
    ):

        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area

    def get_data_as_dataframe(self):

        custom_data_input_dict = {

            "Gender": [self.Gender],
            "Married": [self.Married],
            "Dependents": [self.Dependents],
            "Education": [self.Education],
            "Self_Employed": [self.Self_Employed],
            "ApplicantIncome": [self.ApplicantIncome],
            "CoapplicantIncome": [self.CoapplicantIncome],
            "LoanAmount": [self.LoanAmount],
            "Loan_Amount_Term": [self.Loan_Amount_Term],
            "Credit_History": [self.Credit_History],
            "Property_Area": [self.Property_Area]

        }

        # Create a DataFrame with the same input structure
        # used during model training.
        return pd.DataFrame(custom_data_input_dict)


if __name__ == "__main__":

    # Test prediction directly from Python.

    customer = CustomData(
        Gender="Male",
        Married="Yes",
        Dependents="1",
        Education="Graduate",
        Self_Employed="No",
        ApplicantIncome=5000,
        CoapplicantIncome=1500,
        LoanAmount=120,
        Loan_Amount_Term=360,
        Credit_History=1,
        Property_Area="Urban"
    )

    features = customer.get_data_as_dataframe()

    # Create the prediction pipeline.
    pipeline = PredictPipeline()

    # Generate prediction.
    prediction = pipeline.predict(features)

    # Print the prediction for the single input.
    print(prediction[0])