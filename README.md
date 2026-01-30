News Article Classification using Machine Learning
Project Overview

This project implements an end-to-end machine learning pipeline to classify news articles into predefined categories using Python scripts. The pipeline covers data preprocessing, feature engineering, model training, and evaluation, and is fully runnable from the command line without using Jupyter notebooks.

Dataset Source

BBC News Dataset
Source: Kaggle (Public Dataset)
The dataset contains news articles labeled into categories such as Business, Politics, Sports, Technology, and Entertainment.

Folder Structure Explanation
news_classification_project/
│
├── data/
│   ├── raw/                # Raw dataset (bbc_news.csv)
│   └── processed/          # Processed train/test data and features
│
├── src/
│   ├── data_preprocessing.py   # Text cleaning and train-test split
│   ├── feature_engineering.py  # TF-IDF feature extraction
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
│   └── config.py               # Central configuration file
│
├── models/
│   └── news_classifier.pkl     # Saved trained model
│
├── results/
│   └── metrics.txt             # Accuracy and confusion matrix
│
├── main.py                     # Entry point to run full pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

Steps to Run the Project

Clone or download the repository.

Place the dataset file bbc_news.csv inside:

data/raw/


Install required dependencies:

pip install -r requirements.txt


Run the complete pipeline:

python main.py

Model Used

Logistic Regression for multiclass classification

TF-IDF Vectorization for converting text data into numerical features

This combination is efficient and well-suited for text classification tasks with high-dimensional sparse data.

Final Result Summary

The model successfully classifies news articles into their respective categories.

Final accuracy achieved: 99.10%

The trained model is saved for future use, and evaluation metrics are stored in a results file.