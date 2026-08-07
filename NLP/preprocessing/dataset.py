import pandas as pd


class DatasetLoader:

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None

    # Load Dataset
    def load_dataset(self):
        try:
            self.dataset = pd.read_csv(self.file_path, encoding="utf-8")

            # Remove extra spaces from column names
            self.dataset.columns = self.dataset.columns.str.strip()

            print("Dataset Loaded Successfully!")

            return self.dataset

        except FileNotFoundError:
            print("Error: Dataset file not found.")

        except Exception as e:
            print("Error:", e)

    # Dataset Shape
    def dataset_shape(self):
        print("\nDATASET SHAPE ")
        print(self.dataset.shape)

    # Dataset Columns
    def dataset_columns(self):
        print("\n DATASET COLUMNS ")
        print(self.dataset.columns.tolist())

    # Dataset Information
    def dataset_info(self):
        print("\nDATASET INFO ")
        self.dataset.info()

    # Missing Values
    def missing_values(self):
        print("\nMISSING VALUES  ")
        print(self.dataset.isnull().sum())

    # Duplicate Records
    def duplicate_records(self):
        print("\n DUPLICATE RECORDS ")
        print(self.dataset.duplicated().sum())

    # First Five Rows
    def first_five_rows(self):
        print("\n FIRST FIVE ROWS ")
        print(self.dataset.head())

    # Category Distribution
    def category_distribution(self):
        print("\n  CATEGORY DISTRIBUTION ")
        print(self.dataset["Category"].value_counts())