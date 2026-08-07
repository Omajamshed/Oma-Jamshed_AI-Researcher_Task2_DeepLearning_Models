from sklearn.linear_model import LogisticRegression

class ResumeClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("\n=== MODEL TRAINING COMPLETED =")
        return self.model