from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class FeatureExtractor:

    def __init__(self, dataset):
        self.dataset = dataset

    def extract_features(self):

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )

        X = vectorizer.fit_transform(self.dataset["Resume"])

        # Encode Labels
        encoder = LabelEncoder()

        y = encoder.fit_transform(self.dataset["Category"])

        print("\n=FEATURE EXTRACTION COMPLETED ===")
        print("Feature Matrix Shape :", X.shape)
        print("Target Shape :", y.shape)
        print("Total Classes :", len(encoder.classes_))

        return X, y, vectorizer, encoder