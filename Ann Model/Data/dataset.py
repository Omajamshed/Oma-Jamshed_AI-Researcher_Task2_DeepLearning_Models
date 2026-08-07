import pandas as pd


class DatasetLoader:

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None

    # Load Dataset
    def load_dataset(self):
        try:
            # Load CSV file (comma-separated)
            self.dataset = pd.read_csv(self.file_path)

            # Remove extra spaces from column names
            self.dataset.columns = self.dataset.columns.str.strip()

            print("=" * 50)
            print("Dataset Loaded Successfully!")
            print("=" * 50)

            return self.dataset

        except FileNotFoundError:
            print("Error: Dataset file not found.")
        except Exception as e:
            print("Error while loading dataset:")
            print(e)

    # Dataset Shape
    def dataset_shape(self):
        print("\n========== DATASET SHAPE ==========")
        print(self.dataset.shape)

    # Dataset Columns
    def dataset_columns(self):
        print("\n========== DATASET COLUMNS ==========")
        print(self.dataset.columns.tolist())

    # Dataset Information
    def dataset_info(self):
        print("\n========== DATASET INFORMATION ==========")
        self.dataset.info()

    # Missing Values
    def missing_values(self):
        print("\n========== MISSING VALUES ==========")
        print(self.dataset.isnull().sum())

    # Duplicate Values
    def duplicate_values(self):
        print("\n========== DUPLICATE RECORDS ==========")
        print(self.dataset.duplicated().sum())

    # First Five Rows
    def first_five_rows(self):
        print("\n========== FIRST FIVE ROWS ==========")
        print(self.dataset.head())

    # Statistical Summary
    def statistical_summary(self):
        print("\n========== STATISTICAL SUMMARY ==========")
        print(self.dataset.describe())