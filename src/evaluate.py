import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import PROCESSED_DATA_DIR, MODEL_PATH, METRICS_PATH


def evaluate_model():
    print("Loading model and test data...")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(f"{PROCESSED_DATA_DIR}/X_test_tfidf.pkl", "rb") as f:
        X_test = pickle.load(f)

    with open(f"{PROCESSED_DATA_DIR}/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    print("Evaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Saving evaluation metrics...")

    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    print(f"Evaluation completed. Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    evaluate_model()
