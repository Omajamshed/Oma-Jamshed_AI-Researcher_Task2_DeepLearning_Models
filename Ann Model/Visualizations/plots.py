import matplotlib.pyplot as plt


class TrainingPlots:

    def __init__(self, history):
        self.history = history

    def plot_accuracy(self):

        plt.figure(figsize=(8,5))

        plt.plot(self.history.history["accuracy"], label="Training Accuracy")
        plt.plot(self.history.history["val_accuracy"], label="Validation Accuracy")

        plt.title("ANN Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig("accuracy.png")
        plt.show()

    def plot_loss(self):

        plt.figure(figsize=(8,5))

        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")

        plt.title("ANN Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig("loss.png")
        plt.show()