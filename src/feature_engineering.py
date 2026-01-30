import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import PROCESSED_DATA_DIR


def feature_engineering():
    print("Loading processed text data...")

    with open(f"{PROCESSED_DATA_DIR}/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open(f"{PROCESSED_DATA_DIR}/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)

    print("Applying TF-IDF vectorization...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Saving vectorizer and features...")

    with open(f"{PROCESSED_DATA_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(f"{PROCESSED_DATA_DIR}/X_train_tfidf.pkl", "wb") as f:
        pickle.dump(X_train_tfidf, f)

    with open(f"{PROCESSED_DATA_DIR}/X_test_tfidf.pkl", "wb") as f:
        pickle.dump(X_test_tfidf, f)

    print("Feature engineering completed successfully")


if __name__ == "__main__":
    feature_engineering()
