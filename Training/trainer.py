from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class ModelTrainer:

    def __init__(self, model):
        self.model = model
        self.history = None

    def train(self, X_train, y_train):

        # Stop training if validation loss doesn't improve
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        # Save best model
        checkpoint = ModelCheckpoint(
            "best_ann_model.keras",
            monitor="val_accuracy",
            save_best_only=True
        )

        print("\nTraining Started...\n")

        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        print("\nTraining Completed Successfully!")

        return self.history