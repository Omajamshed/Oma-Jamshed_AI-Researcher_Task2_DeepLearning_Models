import numpy as np
from tensorflow.keras.models import load_model


class WinePredictor:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, features):

        # Convert list into numpy array
        features = np.array(features).reshape(1, -1)

        # Prediction
        prediction = self.model.predict(features)

        if prediction[0][0] >= 0.5:
            print("\nPrediction: Good Quality Wine")
        else:
            print("\nPrediction: Poor Quality Wine")

        print("Prediction Probability:", prediction[0][0])