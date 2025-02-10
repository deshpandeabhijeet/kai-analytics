import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

# Path to save/load the trained model
MODEL_PATH = "./pre-trained-model/visualization_model.joblib"

def train_and_save_model():
    """
    Train the model and save it to disk.
    This should be done offline, not during runtime.
    """
    with open("visualization_data.json", "r") as f:  # Save the JSON data as visualization_data.json
        data = json.load(f)

    df_data = pd.DataFrame()
    all_questions = []
    all_types = []

    for item in data:
      for question in item["questions"]:
        all_questions.append(question)
        all_types.append(item["visualization_type"])

    df_data['questions'] = all_questions
    df_data['visualization_types'] = all_types

    # Train a simple classifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC(kernel='linear'))
    ])

    model.fit(df_data['questions'], df_data['visualization_types'])

    # Save the trained model to disk
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

#train_and_save_model()