import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            "Saved_Models/lstm_model.keras",
            monitor="val_accuracy",
            save_best_only=True
        )

        print("=" * 60)
        print("Training Started...")
        print("=" * 60)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        print("\nTraining Completed Successfully!")

        self.plot_graph(history)

        return history

    def plot_graph(self, history):

        # Accuracy
        plt.figure(figsize=(8,5))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.legend()
        plt.title("Accuracy")
        plt.savefig("Terminal Output/training_accuracy.png")
        plt.close()

        # Loss
        plt.figure(figsize=(8,5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.title("Loss")
        plt.savefig("Terminal Output/training_loss.png")
        plt.close()

        print("Graphs Saved Successfully!")