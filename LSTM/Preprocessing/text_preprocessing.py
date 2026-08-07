import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextPreprocessor:

    def __init__(self, max_words=5000, max_length=100):

        self.max_words = max_words
        self.max_length = max_length

        self.tokenizer = Tokenizer(
            num_words=max_words,
            oov_token="<OOV>"
        )

    def clean_text(self, text):

        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess(self, X_train, X_test):

        print("Text Preprocessing Started...")

        X_train = X_train.apply(self.clean_text)
        X_test = X_test.apply(self.clean_text)

        print("Text Cleaning Completed")

        self.tokenizer.fit_on_texts(X_train)

        train_sequences = self.tokenizer.texts_to_sequences(X_train)
        test_sequences = self.tokenizer.texts_to_sequences(X_test)

        X_train = pad_sequences(
            train_sequences,
            maxlen=self.max_length,
            padding="post"
        )

        X_test = pad_sequences(
            test_sequences,
            maxlen=self.max_length,
            padding="post"
        )

        print("Tokenization Completed")
        print("Padding Completed")

        print("Training Shape :", X_train.shape)
        print("Testing Shape  :", X_test.shape)

        return X_train, X_test, self.tokenizer