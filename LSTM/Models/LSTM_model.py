from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


class LSTMModel:

    def __init__(self,
                 vocab_size=5000,
                 embedding_dim=64,
                 input_length=100):Test-Path ".\LSTM\.git"

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length

    def build_model(self):

        model = Sequential()

        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim
            )
        )

        model.add(
            LSTM(64)
        )

        model.add(Dropout(0.3))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        model.summary()

        return model