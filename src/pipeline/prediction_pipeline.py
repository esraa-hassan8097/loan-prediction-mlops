import pickle  # pickle is Python's built-in library for saving and loading Python objects.
import pandas as pd


class PredictPipeline:

    def __init__(self):
        pass

    def load_objects(self):
        # This method loads the preprocessor and model objects from their respective pickle files.
        # It returns both objects for use in making predictions.

        with open(
            "artifacts/preprocessor.pkl",  # with is used to automatically close the file after the block of code is executed.
            "rb"  # rb means read binary mode. This is necessary because pickle files are binary files.
        ) as file:

            # Read the bytes inside preprocessor.pkl and rebuild the original Python object.
            preprocessor = pickle.load(file)

        with open(
            "artifacts/best_model.pkl",
            "rb"
        ) as file:

            # Read the bytes inside best_model.pkl and rebuild the original Python object
            # (from best_model.pkl back into a LogisticRegression object).
            model = pickle.load(file)

        # Now this function gives us the ColumnTransformer and LogisticRegression objects.
        return (
            preprocessor,
            model
        )

    def predict(self, features):
        # features is the DataFrame returned by customer.get_data_as_dataframe()

        preprocessor, model = self.load_objects()

        # Apply the exact same preprocessing used during training.
        transformed_features = preprocessor.transform(features)

        # This is the line where machine learning actually happens.
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
        # This method belongs to the current CustomData object.
        # When you call:
        # customer.get_data_as_dataframe()
        # Python automatically passes:
        # self = customer

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

        # Now this DataFrame has the same structure as the training data
        # (excluding the target column), so your saved preprocessor
        # can transform it correctly.
        return pd.DataFrame(custom_data_input_dict)


if __name__ == "__main__":

    # Instead of typing into a website,
    # we're entering the values directly in Python.

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

    # This creates an object that knows how to:
    # - Load the preprocessor
    # - Load the model
    # - Make predictions

    pipeline = PredictPipeline()

    prediction = pipeline.predict(features)

    # Instead of printing the whole array,
    # we just print the first element,
    # which is the prediction for our single input data point.
    print(prediction[0])