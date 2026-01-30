from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    print("Starting News Article Classification Pipeline...\n")

    preprocess_data()
    feature_engineering()
    train_model()
    accuracy = evaluate_model()

    print("\nPipeline completed successfully")
    print(f"Final Model Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
