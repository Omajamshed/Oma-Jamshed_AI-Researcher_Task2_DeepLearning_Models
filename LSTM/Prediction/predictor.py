import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


class Predictor:

    def __init__(self, model, tokenizer, max_length=100):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def clean(self, text):

        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z ]', '', text)
        return text

    def predict(self, review):

        review = self.clean(review)
        sequence = self.tokenizer.texts_to_sequences([review])
        sequence = pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding="post"
        )
        prediction = self.model.predict(sequence)[0][0]
        print("\nPrediction Score :", prediction)
        if prediction >= 0.5:
            print("Sentiment : Positive 😊")
        else:
            print("Sentiment : Negative 😞")