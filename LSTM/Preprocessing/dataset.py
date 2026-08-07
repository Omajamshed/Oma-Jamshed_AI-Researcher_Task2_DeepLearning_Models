import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_dataset(self):
        print("Loading IMDB  Dataset.")
        self.data = pd.read_csv(self.file_path)

        print("\nDataset Loaded Successfully!")
        print(f"Shape : {self.data.shape}")

        print("\nFirst Five Rows:")
        print(self.data.head())

        print("\nMissing Values:")
        print(self.data.isnull().sum())

        # Remove Missing Values
        self.data.dropna(inplace=True)

        # Encode Sentiment
        encoder = LabelEncoder()
        self.data["sentiment"] = encoder.fit_transform(self.data["sentiment"])

        print("\nSentiment Encoding:")
        print("positive -> 1")
        print("negative -> 0")

        print("\nClass Distribution:")
        print(self.data["sentiment"].value_counts())

        X = self.data["review"]
        y = self.data["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("\nTrain-Test Split Completed")
        print(f"Training Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")

        return X_train, X_test, y_train, y_test