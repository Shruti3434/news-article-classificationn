import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "bbc_news.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "news_classifier.pkl")

# Results paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")

# ML parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
