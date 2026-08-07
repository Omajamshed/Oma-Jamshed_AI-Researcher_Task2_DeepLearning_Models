# 🍷 ANN-Based White Wine Quality Prediction

## 📌 Project Overview

This project implements an **Artificial Neural Network (ANN)** to predict the quality of white wine using its physicochemical properties. The model is developed using **TensorFlow/Keras** and follows a complete deep learning workflow including data preprocessing, model training, evaluation, and prediction.

The objective is to classify wines into two categories:

- **Good Quality Wine (1)**
- **Poor Quality Wine (0)**

---

# 📂 Project Structure

```
ANN_MODEL/
│
├── Data/
│   ├── dataset.py
│   └── winequalityN.csv
│
├── Preprocessing/
│   └── data_preprocessor.py
│
├── Models/
│   └── ANN_model.py
│
├── Training/
│   └── trainer.py
│
├── Evaluation/
│   └── evaluator.py
│
├── Prediction/
│   └── predict.py
│
├── Graphs/
│   ├── accuracy.png
│   └── loss.png
│
├── best_ann_model.keras
├── main.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

**Dataset:** White Wine Quality Dataset

### Features

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

### Target Variable

Quality

Converted into Binary Classification:

- 1 → Good Quality (Quality ≥ 6)
- 0 → Poor Quality (Quality < 6)

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Missing Value Analysis
- Duplicate Record Detection
- White Wine Filtering
- Binary Target Conversion
- Feature Scaling using StandardScaler
- Train-Test Split
- SMOTE for Class Balancing

---

# 🧠 Artificial Neural Network Architecture

Input Layer

↓

Dense Layer (64 Neurons, ReLU)

↓

Dropout (30%)

↓

Dense Layer (32 Neurons, ReLU)

↓

Dropout (20%)

↓

Dense Layer (16 Neurons, ReLU)

↓

Output Layer (1 Neuron, Sigmoid)

---

# 🚀 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Imbalanced-Learn (SMOTE)

---

# 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training Accuracy Graph
- Validation Accuracy Graph
- Training Loss Graph
- Validation Loss Graph

---

# ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/Omajamshed/Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models.git
```

Move into the project directory

```bash
cd Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

# 📌 Project Workflow

1. Load Dataset
2. Explore Dataset
3. Preprocess Data
4. Build ANN Model
5. Train Model
6. Evaluate Performance
7. Predict Wine Quality

---

# 📷 Output

The project generates:

- Trained ANN Model
- Accuracy Curve
- Loss Curve
- Classification Metrics
- Wine Quality Prediction

---

# 👩‍💻 Author

**Oma Jamshed**

BS Computer Science

University of Karachi

GitHub: https://github.com/Omajamshed

---

# ⭐ Repository

If you found this project useful, consider giving it a ⭐ on GitHub.
