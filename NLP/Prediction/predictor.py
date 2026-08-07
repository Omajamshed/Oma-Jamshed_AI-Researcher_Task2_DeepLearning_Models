import joblib


class ResumePredictor:

    def __init__(self):
        self.model = joblib.load("Saved_Model/resume_classifier.pkl")
        self.vectorizer = joblib.load("Saved_Model/tfidf_vectorizer.pkl")

    def predict_resume(self):
        print("\n== RESUME CATEGORY PREDICTION ==\n")
        resume = input("Paste Resume Text:\n\n")
        resume_vector = self.vectorizer.transform([resume])
        prediction = self.model.predict(resume_vector)[0]
        probability = self.model.predict_proba(resume_vector).max() * 100
        print("\n=== RESULT ===")
        print("Predicted Category :", prediction)
        print(f"Confidence Score : {probability:.2f}%")