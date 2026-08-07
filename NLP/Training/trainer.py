import joblib
import os


class ModelTrainer:

    def __init__(self, model, vectorizer):

        self.model = model
        self.vectorizer = vectorizer

    def save_model(self):

        os.makedirs("Saved_Model", exist_ok=True)

        joblib.dump(self.model, "Saved_Model/resume_classifier.pkl")
        joblib.dump(self.vectorizer, "Saved_Model/tfidf_vectorizer.pkl")

        print("\n== MODEL SAVED SUCCESSFULLY ==")