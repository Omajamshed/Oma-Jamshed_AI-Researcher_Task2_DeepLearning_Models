import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
def evaluate_model(
    model,
    X_test,
    y_test,
    scaler,
    output_dir
):

    print("\nModel Evaluation")
# Predictions
    predictions = model.predict(
        X_test,
        verbose=0
    )
 # Convert back to original passenger values
    predictions_actual = scaler.inverse_transform(
        predictions
    )

    y_actual = scaler.inverse_transform(
        y_test
    )

    # MAE
    mae = mean_absolute_error(
        y_actual,
        predictions_actual
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            predictions_actual
        )
    )

    print("\nEvaluation Results")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # Prediction graph
    plt.figure(figsize=(12, 6))

    plt.plot(
        y_actual,
        label="Actual Passengers",
        linewidth=2
    )

    plt.plot(
        predictions_actual,
        label="RNN Predicted Passengers",
        linewidth=2
    )

    plt.title(
        "RNN Actual vs Predicted Passengers"
    )

    plt.xlabel("Time")
    plt.ylabel("Number of Passengers")

    plt.legend()
    plt.grid(True)

    prediction_path = os.path.join(
        output_dir,
        "rnn_prediction.png"
    )

    plt.savefig(
        prediction_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Prediction graph saved:",
        prediction_path
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Actual": y_actual,
        "Predicted": predictions_actual
    }