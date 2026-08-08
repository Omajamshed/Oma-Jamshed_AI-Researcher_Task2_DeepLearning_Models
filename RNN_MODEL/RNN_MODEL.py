import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
def build_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(
            64,
            activation="tanh",
            input_shape=input_shape,
            return_sequences=False
        ),

        Dropout(0.2),
        Dense(32, activation="relu"),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model