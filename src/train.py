import pickle
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DATA_DIR, MODEL_PATH


def train_model():
    print("Loading TF-IDF features and labels...")

    with open(f"{PROCESSED_DATA_DIR}/X_train_tfidf.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open(f"{PROCESSED_DATA_DIR}/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    print("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Saving trained model...")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model training completed successfully")


if __name__ == "__main__":
    train_model()

