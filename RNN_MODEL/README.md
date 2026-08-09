# 🔄 RNN Air Passenger Forecasting

A complete Deep Learning project implementing a **Recurrent Neural Network (RNN)** for time-series forecasting using the **AirPassengers dataset**.

---

# 📑 Table of Contents

1. Project Overview
2. Objectives
3. What is RNN?
4. Dataset
5. RNN Architecture
6. Architecture Diagram
7. Flow Diagram
8. State Diagram
9. Data Preprocessing
10. Model Details
11. Training Process
12. Evaluation
13. Outputs
14. Folder Structure
15. Installation
16. How to Run
17. Technologies
18. Future Improvements
19. Author

---

# 📌 Project Overview

This project predicts future airline passenger numbers using historical monthly passenger data. A **Simple RNN** learns sequential patterns from the previous 12 months and forecasts the next month's passenger count.

### 🎯 Objectives

* Implement an RNN for time-series forecasting.
* Learn sequential pattern recognition.
* Normalize and preprocess data.
* Generate future passenger predictions.
* Evaluate model performance using MAE & RMSE.
* Save the trained model and visualizations.

---

# 🧠 What is RNN?

A **Recurrent Neural Network (RNN)** is designed for sequential data. Unlike traditional neural networks, it remembers previous information through a **hidden state**, making it ideal for time-series forecasting.

### RNN Working Principle

```text
Month 1 ──►
Month 2 ──►
Month 3 ──►  RNN Hidden State ──► Prediction
...
Month 12 ─►
```

The hidden state carries information from previous months to predict the next value.

---

# 📊 Dataset

**Dataset:** AirPassengers

| Property     | Value           |
| ------------ | --------------- |
| Dataset Type | Time Series     |
| Frequency    | Monthly         |
| Target       | Passenger Count |
| Input Window | 12 Months       |
| Forecast     | Next Month      |

### Example

```text
Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
                     │
                     ▼
              Predict January (Next Year)
```

---

# 🏗️ RNN Architecture

The model uses a Simple RNN followed by Dense layers.

```text
            Input Sequence
        (Previous 12 Months)
                  │
                  ▼
        ┌───────────────────┐
        │    SimpleRNN      │
        │     64 Units      │
        │      tanh         │
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │     Dropout       │
        │       20%         │
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Dense Layer     │
        │     32 Units      │
        │       ReLU        │
        └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐
        │   Output Layer    │
        │      1 Unit       │
        └───────────────────┘
                  │
                  ▼
         Passenger Prediction
```

---

# 🏛️ System Architecture Diagram

```text
┌──────────────────────┐
│ AirPassengers Dataset│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Data Preprocessing   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Sequence Generation  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     RNN Model        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Model Evaluation     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Passenger Forecast   │
└──────────────────────┘
```

---

# 🔄 Project Flow Diagram

┌──────────────────────────────┐
│            START             │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Load Dataset            │
│    AirPassengers.csv         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Data Preprocessing       │
│   Cleaning & Preparation     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Data Scaling           │
│   Normalize Input Values     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Sequence Generation       │
│  Create Input-Output Pairs   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Train/Test Split        │
│   Training & Testing Data    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Build RNN Model        │
│   RNN + Dense + Output       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Train RNN Model        │
│   Learn Temporal Patterns    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Save Trained Model     │
│      rnn_model.keras         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Model Evaluation       │
│       MAE & Performance      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Generate Prediction    │
│    Actual vs Predicted       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Generate Outputs        │
│ Loss • MAE • Prediction      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│             END              │
└──────────────────────────────┘

# 🔁 State Diagram

┌──────────────────────────────┐
│          Input t₁            │
│       Passenger Value        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          RNN Cell            │
│     Process Input t₁         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Hidden State h₁        │
│     Previous Information     │
└──────────────┬───────────────┘
               │
               │
               └──────────────────────┐
                                      │
                                      ▼
                         ┌──────────────────────────────┐
                         │          Input t₂            │
                         │       Passenger Value        │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │          RNN Cell            │
                         │   Input t₂ + Hidden State h₁│
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │       Hidden State h₂        │
                         │   Updated Information        │
                         └──────────────┬───────────────┘
                                        │
                                        │
                                        └─────────────────────┐
                                                              │
                                                              ▼
                                                 ┌──────────────────────┐
                                                 │       Input t₃        │
                                                 │    Passenger Value   │
                                                 └──────────┬───────────┘
                                                            │
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │       RNN Cell        │
                                                 │ Input t₃ + State h₂   │
                                                 └──────────┬───────────┘
                                                            │
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │    Hidden State h₃   │
                                                 │ Updated Information  │
                                                 └──────────┬───────────┘
                                                            │
                                                            ▼
                                                          ...
                                                            │
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │    Final Hidden State│
                                                 │          hₜ          │
                                                 └──────────┬───────────┘
                                                            │
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │      Output Layer    │
                                                 │ Passenger Prediction │
                                                 └──────────────────────┘

# 🧹 Data Preprocessing

The dataset is prepared before training.

### Steps

1. Load AirPassengers dataset
2. Select passenger column
3. Remove missing values
4. Apply Min-Max Scaling
5. Create 12-month sequences
6. Split into training and testing data

### Preprocessing Pipeline

```text
Raw Data
   │
   ▼
Cleaning
   │
   ▼
Scaling
   │
   ▼
Sequence Creation
   │
   ▼
Train/Test Split
```

---

# 🔢 Sequence Creation

The model uses **12 previous months** to predict the next month.

```text
Input:
Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec

Output:
Next Month Passenger Count
```

Sliding window example:

```text
Months 1–12 → Predict Month 13
Months 2–13 → Predict Month 14
Months 3–14 → Predict Month 15
```

---

# ⚙️ Model Details

| Parameter   | Value              |
| ----------- | ------------------ |
| Model       | SimpleRNN          |
| RNN Units   | 64                 |
| Dense Units | 32                 |
| Activation  | tanh + ReLU        |
| Optimizer   | Adam               |
| Loss        | Mean Squared Error |
| Metric      | MAE                |
| Epochs      | 50                 |
| Batch Size  | 16                 |

---

# 🏋️ Model Training

During training, the model learns temporal relationships between historical passenger values.

### Training Pipeline

```text
Training Data
      │
      ▼
Forward Pass
      │
      ▼
Loss Calculation
      │
      ▼
Backpropagation
      │
      ▼
Weight Update
      │
      ▼
Repeat for 50 Epochs
```

---

# 📏 Model Evaluation

The trained model is evaluated using regression metrics.

## MAE

Average absolute prediction error.

```text
Lower MAE = Better Prediction
```

## RMSE

Square root of average squared prediction error.

```text
Lower RMSE = Higher Forecast Accuracy
```

---

# 🔮 Prediction Workflow

```text
Test Sequence
      │
      ▼
Trained RNN
      │
      ▼
Scaled Prediction
      │
      ▼
Inverse Scaling
      │
      ▼
Actual Passenger Value
```

---

# 📈 Outputs

The project automatically generates these files.

```text
Outputs/
│
├── loss.png
├── mae.png
├── rnn_prediction.png
└── rnn_model.keras
```

### Output Description

| File               | Purpose                    |
| ------------------ | -------------------------- |
| loss.png           | Training & Validation Loss |
| mae.png            | Training MAE               |
| rnn_prediction.png | Actual vs Predicted Graph  |
| rnn_model.keras    | Trained Model              |

---

# 📁 Project Structure

```text
RNN_MODEL/
│
├── Data/
│   └── AirPassengers.csv
│
├── Outputs/
│   ├── loss.png
│   ├── mae.png
│   ├── rnn_model.keras
│   └── rnn_prediction.png
│
├── RNN_MODEL.py
├── RNN_Training.py
├── RNN_Evaluation.py
├── RNN_Main.py
│
└── README.md
```

---

# 📂 File Description

### RNN_MODEL.py

* Builds the RNN architecture
* Defines layers
* Compiles the model

### RNN_Training.py

* Loads data
* Preprocesses dataset
* Trains the model
* Saves loss and MAE graphs

### RNN_Evaluation.py

* Loads trained model
* Predicts values
* Calculates MAE & RMSE
* Creates prediction graph

### RNN_Main.py

* Executes the complete project pipeline

---

# 🛠️ Technologies Used

| Technology   | Purpose             |
| ------------ | ------------------- |
| Python       | Programming         |
| TensorFlow   | Deep Learning       |
| Keras        | Neural Networks     |
| Pandas       | Data Processing     |
| NumPy        | Numerical Computing |
| Matplotlib   | Visualization       |
| Scikit-learn | Scaling & Metrics   |

---

# 📦 Required Libraries

```bash
tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
```

Install them using:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

# 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Omajamshed/Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models.git
```

### Open Project

```bash
cd RNN_MODEL
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Run the project using:

```bash
python RNN_Main.py
```

### Execution Pipeline

```text
Load Dataset
      ↓
Preprocess Data
      ↓
Create Sequences
      ↓
Train RNN
      ↓
Evaluate Model
      ↓
Generate Predictions
      ↓
Save Outputs
```

---

# 🎓 Learning Outcomes

After completing this project, you will understand:

* Recurrent Neural Networks
* Sequential Learning
* Time-Series Forecasting
* Sliding Window Technique
* Data Normalization
* Model Evaluation
* Prediction Visualization
* Deep Learning Project Structure

---

# ⚠️ Limitations

* SimpleRNN struggles with long-term dependencies.
* Only one-step forecasting is implemented.
* Performance can improve with LSTM or GRU.

---

# 🚀 Future Improvements

* Implement LSTM
* Implement GRU
* Compare RNN vs LSTM vs GRU
* Multi-step forecasting
* Hyperparameter tuning
* Early stopping
* Streamlit deployment
* FastAPI prediction API

---

# 📌 Project Status

| Component         | Status      |
| ----------------- | ----------- |
| Dataset           | ✅ Completed |
| Preprocessing     | ✅ Completed |
| Sequence Creation | ✅ Completed |
| RNN Model         | ✅ Completed |
| Training          | ✅ Completed |
| Evaluation        | ✅ Completed |
| Prediction        | ✅ Completed |
| Visualizations    | ✅ Completed |
| Saved Model       | ✅ Completed |
| Documentation     | ✅ Completed |

---

# 📝 Conclusion

This project demonstrates a complete implementation of **Recurrent Neural Networks for Time-Series Forecasting**. The AirPassengers dataset is transformed into sequential training samples, processed through a Simple RNN architecture, evaluated using MAE and RMSE, and visualized through prediction graphs. The project provides practical experience in Deep Learning, sequence modeling, and forecasting.

---

# 👩‍💻 Author

**Oma Jamshed**

AI & Deep Learning Researcher

---

⭐ *Part of the Deep Learning Models Repository*
