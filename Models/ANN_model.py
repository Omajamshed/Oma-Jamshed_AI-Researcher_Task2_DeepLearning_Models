from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


class ANNModel:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    # Build ANN Architecture
    def build_model(self):

        self.model = Sequential([

            # Input Layer
            Input(shape=(self.input_shape,)),

            # Hidden Layer 1
            Dense(64, activation="relu"),
            Dropout(0.3),

            # Hidden Layer 2
            Dense(32, activation="relu"),
            Dropout(0.2),

            # Hidden Layer 3
            Dense(16, activation="relu"),

            # Output Layer
            Dense(1, activation="sigmoid")

        ])

        print("\nANN Model Built Successfully!")

        return self.model

    # Compile Model
    def compile_model(self):

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("\nModel Compiled Successfully!")

    # Display Model Summary
    def model_summary(self):

        print("\n========== ANN MODEL SUMMARY ==========\n")
        self.model.summary()