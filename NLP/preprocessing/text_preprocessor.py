import re


class TextPreprocessor:

    def __init__(self, dataset):
        self.dataset = dataset

    # Missing Values
    def check_missing_values(self):
        print("\n==== MISSING VALUES ===")
        print(self.dataset.isnull().sum())

    # Duplicate Records
    def check_duplicates(self):
        duplicates = self.dataset.duplicated().sum()
        print("\nDuplicate Records:", duplicates)

    # Clean Text
    def clean_text(self):

        def preprocess(text):

            text = str(text).lower()

            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            # Remove HTML Tags
            text = re.sub(r'<.*?>', '', text)
            # Remove Punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove Numbers
            text = re.sub(r'\d+', '', text)
            # Remove Extra Spaces
            text = re.sub(r'\s+', ' ', text).strip()

            return text

        self.dataset["Resume"] = self.dataset["Resume"].apply(preprocess)

        print("\nText Cleaning Completed.")

        return self.dataset