
# 🔄 RNN Air Passenger Forecasting

A Deep Learning project that uses a **Recurrent Neural Network (RNN)** to forecast monthly airline passenger numbers using the **AirPassengers dataset**.

---

## 📌 Project Overview

This project implements a **Simple Recurrent Neural Network (RNN)** for time-series forecasting.

The model learns patterns from historical airline passenger data and uses the previous **12 months** of passenger information to predict the passenger count for the next month.

### Project Objectives

- Understand Recurrent Neural Networks
- Work with sequential time-series data
- Preprocess and scale the dataset
- Create time-series sequences
- Train an RNN model
- Evaluate forecasting performance
- Generate future predictions
- Visualize actual vs predicted values
- Save the trained Keras model

---

# 🧠 RNN Architecture

The RNN architecture used in this project is:

```text
              Input Sequence
             Previous 12 Months
                    │
                    ▼
        ┌──────────────────────┐
        │      SimpleRNN       │
        │      64 Units        │
        │   tanh Activation    │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │       Dropout        │
        │        20%           │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │     Dense Layer      │
        │      32 Units        │
        │   ReLU Activation    │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │     Output Layer     │
        │        1 Unit        │
        └──────────────────────┘
                    │
                    ▼
          Passenger Prediction



Architecture Explanation

1. Input Sequence

The model receives the previous 12 months of passenger data.

2. SimpleRNN

The SimpleRNN layer contains 64 units and learns temporal patterns from the input sequence.

3. Dropout

A dropout rate of 20% is used to reduce overfitting.

4. Dense Layer

A fully connected layer with 32 neurons processes the learned representation.

5. Output Layer

The final layer contains 1 neuron because the model predicts one passenger value for the next time step.


🔄 Flow Diagram

The complete workflow of the project is shown below:
flowchart TD

    A([Start]) --> B[Load AirPassengers Dataset]

    B --> C[Data Cleaning]

    C --> D[Select Passenger Column]

    D --> E[Min-Max Scaling]

    E --> F[Create 12-Month Sequences]

    F --> G[Train/Test Split]

    G --> H[Build RNN Model]

    H --> I[Train RNN]

    I --> J[Save Training Loss]

    I --> K[Save Training MAE]

    I --> L[Save Trained Model]

    I --> M[Generate Predictions]

    M --> N[Inverse Scaling]

    N --> O[Calculate MAE]

    N --> P[Calculate RMSE]

    O --> Q[Actual vs Predicted Graph]

    P --> Q

    Q --> R[Save Prediction Graph]

    R --> S([Project Completed])

🔁 State Diagram

The system moves through different states during execution.

stateDiagram-v2

    [*] --> Initialization

    Initialization --> DataLoading

    DataLoading --> DataValidation

    DataValidation --> ErrorState: Dataset Missing

    ErrorState --> [*]

    DataValidation --> Preprocessing: Dataset Valid

    Preprocessing --> SequenceCreation

    SequenceCreation --> TrainTestSplit

    TrainTestSplit --> ModelBuilding

    ModelBuilding --> Training

    Training --> ModelSaving

    ModelSaving --> Prediction

    Prediction --> Evaluation

    Evaluation --> Visualization

    Visualization --> Completed

    Completed --> [*]

