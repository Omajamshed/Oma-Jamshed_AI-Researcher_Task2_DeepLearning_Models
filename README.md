# 🧠 Deep Learning Models — AI Researcher Task 2

A complete Deep Learning repository containing practical implementations of different neural network architectures and AI techniques using **Python, TensorFlow, Keras, Scikit-learn, and NLP tools**.

The repository demonstrates how different models are applied to different types of data, including **tabular data, images, text, and time-series sequences**.

---

## 📌 Projects Included

| Project              | Model / Technique           | Dataset                  | Task                        |
| -------------------- | --------------------------- | ------------------------ | --------------------------- |
| 🧠 ANN               | Artificial Neural Network   | White Wine Quality       | Binary Classification       |
| 🔄 RNN               | Recurrent Neural Network    | AirPassengers            | Time-Series Forecasting     |
| 🧠 LSTM              | Long Short-Term Memory      | IMDB Dataset             | Sentiment Classification    |
| 📝 NLP               | Natural Language Processing | Resume Screening Dataset | Resume Classification       |
| 🌸 Transfer Learning | MobileNetV2                 | TF Flowers               | Flower Image Classification |

> **CNN:** CNN concepts are demonstrated through the Transfer Learning project, where **MobileNetV2**, a pretrained CNN architecture, is used for image classification.

---

# 🧠 1. ANN — Artificial Neural Network

## 📌 Overview

The ANN project predicts **white wine quality** using physicochemical properties of wine.

The original quality score is converted into a binary classification problem:

```text
Quality >= 6  →  Good Quality (1)

Quality < 6   →  Poor Quality (0)
```

### 📊 Dataset

The project uses the **White Wine Quality Dataset**.

Main features include:

* Fixed Acidity
* Volatile Acidity
* Citric Acid
* Residual Sugar
* Chlorides
* Free Sulfur Dioxide
* Total Sulfur Dioxide
* Density
* pH
* Sulphates
* Alcohol

### 🔄 ANN Workflow

```text
┌──────────────────────┐
│   Wine Dataset       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Data Preprocessing   │
│ Cleaning & Filtering │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Binary Classification│
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ StandardScaler       │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ SMOTE Balancing      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ ANN Model            │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Training & Evaluation│
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Wine Quality         │
│ Prediction           │
└──────────────────────┘
```

### 🏗️ ANN Architecture

```text
Input Features
      ↓
Dense — 64 Neurons
      ↓
ReLU
      ↓
Dropout — 30%
      ↓
Dense — 32 Neurons
      ↓
ReLU
      ↓
Dropout — 20%
      ↓
Dense — 16 Neurons
      ↓
ReLU
      ↓
Output — 1 Neuron
      ↓
Sigmoid
      ↓
Good / Poor Quality
```

### 📈 Evaluation

The ANN project evaluates:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

---

# 🔄 2. RNN — Recurrent Neural Network

## 📌 Overview

The RNN project performs **air passenger forecasting** using historical monthly passenger data.

An RNN is suitable for sequential data because it maintains information from previous time steps through its hidden state.

### 📊 Dataset

**AirPassengers Dataset**

| Property     | Details            |
| ------------ | ------------------ |
| Dataset Type | Time Series        |
| Frequency    | Monthly            |
| Target       | Passenger Count    |
| Input Window | Previous 12 Months |
| Prediction   | Next Month         |

### 🔄 RNN Workflow

```text
┌────────────────────────┐
│ AirPassengers Dataset  │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Data Preprocessing     │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Data Scaling           │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Sequence Generation    │
│ Previous 12 Months     │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Train/Test Split       │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Simple RNN             │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Prediction             │
└────────────────────────┘
```

### 🏗️ RNN Architecture

```text
Previous 12 Months
        ↓
┌──────────────────┐
│ SimpleRNN        │
│ 64 Units         │
│ tanh             │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Dropout — 20%    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Dense — 32        │
│ ReLU              │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Output — 1        │
└────────┬─────────┘
         ↓
Passenger Forecast
```

### 📊 Evaluation

The RNN project uses:

* MAE
* RMSE
* Actual vs Predicted Visualization
* Forecast Visualization

---

# 🧠 3. LSTM — Long Short-Term Memory

## 📌 Overview

The LSTM project performs **sentiment classification on movie reviews** using the IMDB dataset.

LSTM is an advanced form of RNN designed to learn long-term dependencies in sequential data.

### 📊 Dataset

**IMDB Movie Reviews Dataset**

The dataset contains movie reviews with sentiment labels.

```text
Review
   ↓
Positive / Negative Sentiment
```

### 🔄 LSTM Workflow

```text
┌────────────────────────┐
│ IMDB Movie Reviews     │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Text Preprocessing     │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Tokenization           │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Sequence Padding       │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ LSTM Model             │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Training               │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Evaluation             │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Sentiment Prediction   │
└────────────────────────┘
```

### 🔐 LSTM Concept

LSTM uses memory mechanisms and gates to control information:

```text
Input Sequence
      ↓
┌───────────────────┐
│ LSTM Memory Cell  │
│                   │
│ Forget Gate       │
│ Input Gate        │
│ Output Gate       │
└─────────┬─────────┘
          ↓
Sentiment Prediction
```

### 🎯 Prediction Example

```text
"This movie was absolutely amazing.
I loved every scene."

                ↓

        Positive Sentiment
```

### 📈 Evaluation

* Accuracy
* Loss
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

---

# 📝 4. NLP — Resume Screening & Classification

## 📌 Overview

The NLP project implements an **NLP-based Resume Screening and Classification system**.

The system processes resume text and predicts the most suitable resume category.

Unlike ANN, CNN, RNN, or LSTM, **NLP is a field rather than a single neural-network architecture**. This project demonstrates practical NLP preprocessing and text classification using TF-IDF and Logistic Regression.

### 📊 Dataset

**Resume Screening Dataset**

Main columns:

| Column   | Description     |
| -------- | --------------- |
| Resume   | Resume text     |
| Category | Resume category |

### 🔄 NLP Workflow

```text
┌────────────────────────┐
│ Resume Dataset         │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Text Cleaning          │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Token / Text Processing│
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ TF-IDF Feature         │
│ Extraction             │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Label Encoding         │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Train/Test Split       │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Logistic Regression    │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Evaluation             │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ New Resume Prediction  │
└────────────────────────┘
```

### 🧹 Text Preprocessing

The project performs:

* Lowercase conversion
* URL removal
* HTML removal
* Punctuation removal
* Number removal
* Extra-space removal

### 🔤 TF-IDF

TF-IDF converts resume text into numerical features.

Configuration used in the project:

```text
stop_words = English
max_features = 5000
```

### 🤖 Classification Model

The processed text is classified using:

```text
LogisticRegression(max_iter=1000)
```

### 📈 Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* Prediction Confidence

---

# 🌸 5. Transfer Learning — Flower Classification

## 📌 Overview

The Transfer Learning project performs **flower image classification using MobileNetV2**.

MobileNetV2 is a pretrained CNN architecture. The model uses knowledge learned from ImageNet and adapts it to the TF Flowers dataset.

### 📊 Dataset

**TF Flowers Dataset**

Approximately **3,670 RGB images** belonging to five classes.

```text
┌─────────────────────────┐
│     TF Flowers          │
└────────────┬────────────┘
             │
     ┌───────┼─────────────┐
     ↓       ↓       ↓      ↓
   Daisy  Dandelion Roses  Sunflowers
                            ↓
                          Tulips
```

### 🔄 Transfer Learning Workflow

```text
┌────────────────────────┐
│ TF Flowers Dataset     │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Image Preprocessing    │
│ 224 × 224 × 3          │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Data Augmentation       │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Pretrained MobileNetV2 │
│ ImageNet Weights       │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Feature Extraction     │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Custom Classification  │
│ Head                   │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Initial Training       │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Fine-Tuning             │
└───────────┬────────────┘
            ↓
┌────────────────────────┐
│ Flower Prediction      │
└────────────────────────┘
```

### 🧠 MobileNetV2 Architecture

```text
Input Image
     ↓
MobileNetV2
     ↓
Feature Extraction
     ↓
Global Average Pooling
     ↓
Dense Layer
     ↓
Dropout
     ↓
Softmax
     ↓
5 Flower Classes
```

### 🔥 Fine-Tuning

The project uses two stages:

**Stage 1 — Feature Extraction**

```text
MobileNetV2 Base → Frozen
Classification Head → Trainable
```

**Stage 2 — Fine-Tuning**

```text
Selected MobileNetV2 Layers → Unfrozen
          ↓
Lower Learning Rate
          ↓
Fine-Tuning
```

### 📈 Evaluation

* Accuracy
* Loss
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix
* Prediction Visualization

---

# 🖼️ CNN in This Repository

CNN is represented through the **Transfer Learning project**, where MobileNetV2 is used as the pretrained CNN backbone.

CNNs are designed for image data and learn visual features hierarchically:

```text
Image
  ↓
Edges
  ↓
Textures
  ↓
Shapes
  ↓
Complex Features
  ↓
Object / Flower Class
```

Therefore:

```text
CNN
 ↓
MobileNetV2
 ↓
Transfer Learning
 ↓
Flower Classification
```

---

# 📊 Complete Model Comparison

| Model / Technique | Data            | Main Task      | Main Strength                                | Main Limitation                           |
| ----------------- | --------------- | -------------- | -------------------------------------------- | ----------------------------------------- |
| ANN               | Tabular         | Classification | Good for structured numerical data           | Does not model spatial/temporal structure |
| CNN               | Images          | Classification | Learns spatial/visual features               | Mainly suited to grid-like data           |
| RNN               | Time Series     | Forecasting    | Processes sequential information             | Limited long-term memory                  |
| LSTM              | Text / Sequence | Classification | Learns long-term dependencies                | More complex than basic RNN               |
| NLP               | Text            | Classification | Converts human language into useful features | Requires text preprocessing               |
| Transfer Learning | Images          | Classification | Reuses pretrained knowledge                  | Depends on suitable pretrained model      |

---

# ⚙️ Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Machine Learning

* Scikit-learn
* Imbalanced-learn

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### NLP

* TF-IDF
* Text preprocessing
* Tokenization
* Label Encoding

### Development Tools

* PyCharm
* Jupyter Notebook
* Google Colab
* Git
* GitHub

---

# 🔄 Overall Repository Workflow

```text
                    ┌──────────────┐
                    │    DATA      │
                    └──────┬───────┘
                           ↓
                 ┌──────────────────┐
                 │  PREPROCESSING   │
                 └────────┬─────────┘
                          ↓
              ┌────────────────────────┐
              │ MODEL / TECHNIQUE      │
              └───────────┬────────────┘
                          ↓
                 ┌──────────────────┐
                 │    TRAINING      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   EVALUATION     │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   PREDICTION     │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │     OUTPUTS      │
                 └──────────────────┘
```

---

# 📁 Repository Structure

```text
Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models/
│
├── ANN_MODEL/
│
├── Ann Model/
│
├── RNN_MODEL/
│
├── LSTM/
│
├── NLP/
│
├── Transformlearning/
│
├── .idea/
│
├── main.py
│
├── .gitignore
│
└── README.md
```

### Project Organization

Each major project contains its own implementation files, datasets, preprocessing components, model/training code, evaluation logic, prediction functionality, visualizations, saved models, or output files according to the requirements of that project.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Omajamshed/Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models.git
```

Move into the repository:

```bash
cd Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages for the selected project:

```bash
pip install -r requirements.txt
```

Some projects contain their own `requirements.txt`, so dependencies should be installed from the respective project folder when required.

---

# ▶️ How to Run

Navigate to the project you want to execute.

### ANN

```bash
cd "Ann Model"
python main.py
```

### RNN

```bash
cd RNN_MODEL
python RNN_Main.py
```

### LSTM

```bash
cd LSTM
python main.py
```

### NLP

```bash
cd NLP
python main.py
```

### Transfer Learning

The Transfer Learning project is provided as a Jupyter Notebook:

```text
Transformlearning/
└── Transferlearning.ipynb
```

It can be opened using **Jupyter Notebook, JupyterLab, or Google Colab**.

---

# 📈 Evaluation Metrics

Different projects use metrics according to their task.

### Classification Projects

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

### Regression / Forecasting

* MAE
* RMSE
* Actual vs Predicted Values
* Forecast Visualization

### Training Analysis

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

---

# 📦 Outputs

The projects generate different outputs depending on their task, including:

```text
✓ Trained Models
✓ Saved Models
✓ Accuracy Graphs
✓ Loss Graphs
✓ Confusion Matrices
✓ Classification Reports
✓ Prediction Results
✓ Forecast Graphs
✓ Visualization Outputs
✓ Model Evaluation Results
```

---

# 🎓 Learning Outcomes

This repository demonstrates practical understanding of:

* Artificial Neural Networks
* Convolutional Neural Networks
* Recurrent Neural Networks
* LSTM Networks
* Natural Language Processing
* Transfer Learning
* Text Preprocessing
* Image Preprocessing
* Feature Scaling
* TF-IDF
* Tokenization
* Sequence Generation
* Data Augmentation
* SMOTE
* Model Training
* Model Evaluation
* Model Prediction
* Fine-Tuning
* Model Saving

---

# 🏆 Key Takeaway

Different Deep Learning approaches are suitable for different types of problems:

```text
Tabular Data
      ↓
     ANN

Image Data
      ↓
     CNN

Time-Series / Sequential Data
      ↓
     RNN

Long-Term Sequential Dependencies
      ↓
    LSTM

Text / Language
      ↓
     NLP

Image Classification + Pretrained Knowledge
      ↓
Transfer Learning
      ↓
   MobileNetV2
```

The repository therefore provides a practical comparison of **traditional neural networks, convolutional models, recurrent architectures, NLP techniques, and pretrained Deep Learning models**.

---

# 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Early stopping
* Learning-rate scheduling
* GRU implementation
* Bidirectional LSTM
* Transformer-based NLP
* BERT-based classification
* ResNet and EfficientNet
* Additional CNN architectures
* Model deployment using Streamlit
* REST API deployment using FastAPI
* Real-time prediction applications

---

# 👩‍💻 Author

**Oma Jamshed**

BS Computer Science
University of Karachi

GitHub: **Omajamshed**

---

# ⭐ Conclusion

This repository is a practical Deep Learning portfolio demonstrating multiple approaches for solving problems involving **tabular data, images, text, and sequential data**.

It includes an ANN for **wine quality classification**, an RNN for **air passenger forecasting**, an LSTM for **IMDB sentiment classification**, an NLP pipeline for **resume screening and classification**, and a MobileNetV2-based Transfer Learning project for **flower image classification**.

Together, these projects demonstrate the complete Deep Learning workflow:

```text
Data
 ↓
Preprocessing
 ↓
Model Building
 ↓
Training
 ↓
Evaluation
 ↓
Prediction
 ↓
Visualization
 ↓
Saved Model
```

**Built with Python, TensorFlow, Keras, Scikit-learn, and NLP techniques.**
