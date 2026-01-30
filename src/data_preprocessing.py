import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    TEST_SIZE,
    RANDOM_STATE
)   


def clean_text(text: str) -> str:
    """
    Clean news article text:
    - lowercase
    - remove special characters & numbers
    - remove stopwords
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)


def preprocess_data():
    print(" Loading dataset...")
    df = pd.read_csv(
    RAW_DATA_PATH,
    sep="\t",
    engine="python",
    on_bad_lines="skip"
)


    print(" Columns found:", df.columns.tolist())

    # Keep only required columns
    df = df[["content", "category"]]

    # Drop missing values
    df.dropna(inplace=True)

    print(" Cleaning text data...")
    df["content"] = df["content"].apply(clean_text)

    X = df["content"]
    y = df["category"]

    print(" Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(" Saving processed data...")
    with open(f"{PROCESSED_DATA_DIR}/X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)

    with open(f"{PROCESSED_DATA_DIR}/X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)

    with open(f"{PROCESSED_DATA_DIR}/y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)

    with open(f"{PROCESSED_DATA_DIR}/y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)

    print(" Data preprocessing completed successfully")
