import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class Evaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):

        print("Model Evaluation")

        predictions = self.model.predict(X_test)

        predictions = (predictions > 0.5).astype(int)

        accuracy = accuracy_score(y_test, predictions)

        print(f"\nAccuracy : {accuracy:.4f}")

        print("\nClassification Report\n")
        print(classification_report(y_test, predictions))

        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(6,5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative","Positive"],
            yticklabels=["Negative","Positive"]
        )

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig("Terminal Output/confusion_matrix.png")

        plt.close()

        print("\nConfusion Matrix Saved Successfully!")