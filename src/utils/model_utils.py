from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train a model and evaluate its performance.

    Parameters
    ----------
    model : sklearn estimator
        The machine learning model.
    X_train, y_train :
        Training data.
    X_test, y_test :
        Testing data.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            pos_label="Y"
        ),
        "recall": recall_score(
            y_test,
            predictions,
            pos_label="Y"
        ),
        "f1": f1_score(
            y_test,
            predictions,
            pos_label="Y"
        ),
    }

    return metrics