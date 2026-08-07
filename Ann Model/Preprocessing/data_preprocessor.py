from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class DataPreprocessor:

    def __init__(self, dataset):
        self.dataset = dataset
        self.X = None
        self.y = None

    # Missing Values
    def check_missing_values(self):
        print("\n=== Missing Values =====")
        print(self.dataset.isnull().sum())

    # Duplicate Values
    def check_duplicates(self):
        duplicates = self.dataset.duplicated().sum()
        print("\nDuplicate Records:", duplicates)

    def prepare_dataset(self):
        # Keeping  only White Wine
        self.dataset = self.dataset[self.dataset["type"] == "white"]

        # Remove type column
        self.dataset.drop("type", axis=1, inplace=True)

        # Remove Missing Values
        self.dataset.dropna(inplace=True)

        # Remove Duplicates
        self.dataset.drop_duplicates(inplace=True)

        # Convert Quality into Binary
        self.dataset["quality"] = self.dataset["quality"].apply(
            lambda x: 1 if x >= 6 else 0
        )

        print("\nDataset Shape After Cleaning:")
        print(self.dataset.shape)

        print("\nTarget Distribution:")
        print(self.dataset["quality"].value_counts())
    # Features & Target
    def split_features_target(self):
        self.X = self.dataset.drop("quality", axis=1)
        self.y = self.dataset["quality"]

        print("\nFeatures Shape :", self.X.shape)
        print("Target Shape   :", self.y.shape)
    # Feature Scaling
    def feature_scaling(self):
        scaler = StandardScaler()

        self.X = scaler.fit_transform(self.X)

        print("\nFeature Scaling Completed.")
    # Train Test Split
    def train_test_split_data(self):
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.20,
            random_state=42,
            stratify=self.y
        )

        # Apply SMOTE ONLY on Training Data
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("\nTrain-Test Split Completed.")
        print("\nBalanced Training Target Distribution:")
        print(y_train.value_counts())
        print("\nTesting Target Distribution:")
        print(y_test.value_counts())

        return X_train, X_test, y_train, y_test

    # Complete Preprocessing
    def preprocess(self):

        self.check_missing_values()
        self.check_duplicates()
        self.prepare_dataset()
        self.split_features_target()
        self.feature_scaling()

        return self.train_test_split_data()