from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):

        # Predict Probabilities
        y_pred_prob = self.model.predict(X_test)

        # Convert probabilities into binary values
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        print(f"\nTest Accuracy : {accuracy:.4f}")

        # Confusion Matrix
        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        # Classification Report
        print("\nClassification Report")
        print(classification_report(y_test, y_pred))