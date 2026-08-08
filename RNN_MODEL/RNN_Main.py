import os

from RNN_Training import (
    load_and_preprocess_data,
    train_rnn
)
from RNN_Evaluation import evaluate_model
def main():

    print("RNN AIR PASSENGER FORECASTING")
 # Current RNN_MODEL folder
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )
# Dataset
    data_path = os.path.join(
        base_dir,
        "Data",
        "AirPassengers.csv"
    )

    # Outputs
    output_dir = os.path.join(
        base_dir,
        "Outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Check dataset
    if not os.path.exists(data_path):

        print("\nERROR!")
        print(
            "AirPassengers.csv not found."
        )
        print(
            "Expected location:"
        )
        print(data_path)

        return
    print("\nDataset found successfully!")

    # PREPROCESSING
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = load_and_preprocess_data(
        data_path,
        sequence_length=12
    )

    # TRAINING

    model, history = train_rnn(
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir,
        epochs=50,
        batch_size=16
    )

    # EVALUATION

    results = evaluate_model(
        model,
        X_test,
        y_test,
        scaler,
        output_dir
    )

    # FINAL RESULTS

    print("FINAL RESULTS")
    print(
        f"MAE  : {results['MAE']:.2f}"
    )
    print(
        f"RMSE : {results['RMSE']:.2f}"
    )
    print("\nGenerated Outputs:")

    print(
        "loss.png"
    )

    print(
        "mae.png"
    )

    print(
        "rnn_model.keras"
    )

    print(
        "rnn_prediction.png"
    )

    print("\nRNN PROJECT COMPLETED!")


if __name__ == "__main__":
    main()