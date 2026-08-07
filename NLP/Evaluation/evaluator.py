from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
class Evaluator:
     def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

     def evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print("\nMODEL EVALUATION")
        print(f"\nAccuracy : {accuracy:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(self.y_test, predictions))
        cm = confusion_matrix(self.y_test, predictions)
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.model.classes_
        )

        display.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("Visualizations/confusion_matrix.png")
        plt.show()