import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from RNN_MODEL import build_rnn_model
def load_and_preprocess_data(data_path, sequence_length=12):

    print("\nLoading AirPassengers Dataset")

    df = pd.read_csv(data_path)
    # dataset shape
    print("Dataset Shape:", df.shape)
    # column
    print("Columns:", df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    # Find passenger column
    if "Passengers" in df.columns:
        passenger_column = "Passengers"
    elif "passengers" in df.columns:
        passenger_column = "passengers"
    else:
        # Use last numeric column
        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) == 0:
            raise ValueError(
                "No numeric passenger column found."
            )

        passenger_column = numeric_columns[-1]

    values = df[passenger_column].values.astype(
        "float32"
    ).reshape(-1, 1)

    print("\nTarget Column:", passenger_column)
    print("Total Records:", len(values))

    # Scaling
    scaler = MinMaxScaler()

    scaled_values = scaler.fit_transform(values)

    # Create sequences
    X = []
    y = []

    for i in range(sequence_length, len(scaled_values)):

        X.append(
            scaled_values[
                i - sequence_length:i
            ]
        )

        y.append(
            scaled_values[i]
        )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Chronological split
    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    print("\nPreprocessing Completed")
    print("X Train:", X_train.shape)
    print("X Test :", X_test.shape)
    print("y Train:", y_train.shape)
    print("y Test :", y_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )


def train_rnn(
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir,
    epochs=50,
    batch_size=16
):

    print("\nBuilding RNN Model")

    model = build_rnn_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    model.summary()

    print("\n RNN Training Started")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_test,
            y_test
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    print("\nTraining Completed")

    # Save model
    model_path = os.path.join(
        output_dir,
        "rnn_model.keras"
    )

    model.save(model_path)

    print("Model saved:", model_path)

    # Loss graph
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("RNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(
        output_dir,
        "loss.png"
    )

    plt.savefig(loss_path)
    plt.close()

    print("Loss graph saved:", loss_path)

    # MAE graph
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["mae"],
        label="Training MAE"
    )

    plt.plot(
        history.history["val_mae"],
        label="Validation MAE"
    )

    plt.title("RNN Training and Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)

    mae_path = os.path.join(
        output_dir,
        "mae.png"
    )

    plt.savefig(mae_path)
    plt.close()

    print("MAE graph saved:", mae_path)

    return model, history