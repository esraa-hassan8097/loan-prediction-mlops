from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.utils.model_utils import evaluate_model


def test_evaluate_model_returns_metrics():

    # Create a small synthetic dataset
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        random_state=42
    )

    # Convert labels to match your project
    y = ["Y" if label == 1 else "N" for label in y]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create model
    model = LogisticRegression()

    # Evaluate model
    metrics = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Verify returned dictionary
    assert isinstance(metrics, dict)

    # Verify required metrics exist
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    # Verify metric ranges
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1