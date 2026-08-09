# 🧠 NLP Resume Screening & Classification

## 📌 Project Overview

This project is an **NLP-based Resume Screening and Classification system** that automatically analyzes resume text and predicts the most suitable resume category.

The project implements a complete Natural Language Processing pipeline:

**Dataset Loading → Text Preprocessing → TF-IDF Feature Extraction → Train/Test Split → Logistic Regression → Evaluation → Model Saving → Resume Prediction**

The system takes raw resume text as input, converts the text into numerical features using **TF-IDF**, and uses a **Logistic Regression classifier** to predict the resume category.

---

# 🎯 Objectives

The main objectives of this project are:

* Load and explore resume screening data.
* Analyze missing values and duplicate records.
* Clean and preprocess resume text.
* Visualize resume category distribution.
* Generate a word cloud from resume text.
* Convert textual data into numerical features using TF-IDF.
* Encode resume categories into numerical labels.
* Train a Logistic Regression classifier.
* Evaluate the classification model.
* Generate a confusion matrix.
* Save the trained model and TF-IDF vectorizer.
* Predict the category of a new resume.
* Display the prediction confidence score.

---

# 🧠 What is NLP?

**Natural Language Processing (NLP)** is a branch of Artificial Intelligence that enables computers to process and analyze human language.

In this project, NLP is used to process **resume text** and extract useful textual features that can be used for automatic resume classification.

The system transforms unstructured resume text into numerical features and then uses those features for classification.

---

# 📊 Dataset

The project uses the:

### **Resume Screening Dataset**

Dataset file:

```text
Data/Resume Screening.csv
```

The dataset contains resume text and its corresponding category.

The main columns used by the project are:

| Column     | Description              |
| ---------- | ------------------------ |
| `Resume`   | Contains the resume text |
| `Category` | Target resume category   |

The dataset file is stored inside the `Data` folder of the project.

---

# 🔄 Complete Project Workflow

```text
┌────────────────────────────┐
│     Resume Dataset         │
│  Resume + Category         │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│     Data Exploration       │
│ Shape / Columns / Missing  │
│ Values / Duplicates        │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│    Text Preprocessing      │
│ Cleaning Resume Text       │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│   Feature Extraction       │
│       TF-IDF              │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│     Label Encoding         │
│ Category → Numerical Label │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│      Train/Test Split      │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│    Logistic Regression     │
│        Classifier          │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│       Evaluation           │
│ Accuracy + Report + CM     │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│       Save Model           │
│ Model + TF-IDF Vectorizer  │
└──────────────┬─────────────┘
               ↓
┌────────────────────────────┐
│    New Resume Prediction   │
└────────────────────────────┘
```

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Resume Screening   │
                    │       Dataset        │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Text Preprocessor  │
                    │                      │
                    │ • Lowercase          │
                    │ • Remove URLs        │
                    │ • Remove HTML        │
                    │ • Remove punctuation │
                    │ • Remove numbers     │
                    │ • Remove extra space │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   TF-IDF Vectorizer  │
                    │                      │
                    │ max_features = 5000  │
                    │ stop_words = English │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Label Encoder     │
                    │   Category → Label   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Train/Test Split  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Logistic Regression  │
                    │      Classifier      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │      Evaluation      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │     Prediction       │
                    └──────────────────────┘
```

---

# 🔄 Flow Diagram

```text
┌───────────────┐
│     START     │
└───────┬───────┘
        ↓
┌────────────────────┐
│  Load Resume Data  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Explore Dataset    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Clean Resume Text  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Generate Wordcloud │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ TF-IDF Extraction  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Encode Categories  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Train/Test Split   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Train Logistic     │
│ Regression Model   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Evaluate Model     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Save Model         │
│ + TF-IDF Vectorizer│
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Predict New Resume │
└─────────┬──────────┘
          ↓
┌───────────────┐
│      END      │
└───────────────┘
```

---

# 🔁 State Diagram

```text
┌─────────────────────┐
│   DATASET STATE     │
│ Raw Resume Dataset  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ PREPROCESSING STATE │
│   Cleaned Resume    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ FEATURE STATE       │
│      TF-IDF         │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ ENCODING STATE      │
│ Category → Labels   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ TRAINING STATE      │
│ Logistic Regression │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ EVALUATION STATE    │
│ Accuracy + Report   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ SAVING STATE        │
│ Model + Vectorizer  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ PREDICTION STATE    │
│ New Resume Input    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ OUTPUT STATE        │
│ Category + Confidence│
└─────────────────────┘
```

---

# 🧹 Text Preprocessing

The `TextPreprocessor` class cleans the `Resume` column before feature extraction.

The following operations are performed:

### 1. Lowercase Conversion

All resume text is converted to lowercase.

```text
"Software Engineer"
        ↓
"software engineer"
```

### 2. URL Removal

URLs such as:

```text
https://example.com
www.example.com
```

are removed.

### 3. HTML Tag Removal

HTML tags are removed from the resume text.

### 4. Punctuation Removal

Unnecessary punctuation is removed.

### 5. Number Removal

Numeric characters are removed.

### 6. Extra Space Removal

Multiple spaces are converted into a single space.

These preprocessing operations are implemented in `preprocessing/text_preprocessor.py`.

---

# 🔤 Feature Extraction Using TF-IDF

After preprocessing, the resume text is converted into numerical features using:

### **TF-IDF — Term Frequency-Inverse Document Frequency**

The project uses:

```text
TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
```

This means:

* English stop words are removed.
* A maximum of 5,000 features is considered.
* Resume text is transformed into a numerical feature matrix.

The TF-IDF representation is generated from the `Resume` column.

---

# 🏷️ Label Encoding

The target column is:

```text
Category
```

The category values are converted into numerical labels using `LabelEncoder`.

```text
Resume Category
       ↓
Label Encoder
       ↓
Numerical Class
```

This allows the classification model to work with the category labels.

---

# 🧠 Model Details

## Logistic Regression Classifier

The actual classifier implemented in this project is:

**Logistic Regression**

with:

```python
LogisticRegression(max_iter=1000)
```

The model is trained using the TF-IDF feature matrix and encoded category labels.

### Model Architecture

Because this project uses Logistic Regression rather than a neural network, its architecture is represented as:

```text
Resume Text
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Numerical Feature Vector
     ↓
Logistic Regression
     ↓
Predicted Resume Category
```

---

# ⚙️ Training

The training pipeline performs the following steps:

```text
TF-IDF Features
       ↓
Train/Test Split
       ↓
X_train + y_train
       ↓
Logistic Regression
       ↓
Model Fitting
       ↓
Trained Classifier
```

The `trainer.py` module is responsible for saving the trained classifier and its TF-IDF vectorizer.

---

# 📈 Model Evaluation

The project evaluates the trained classifier using:

* Accuracy
* Classification Report
* Confusion Matrix

The evaluation module uses:

```text
accuracy_score
classification_report
confusion_matrix
ConfusionMatrixDisplay
```

The confusion matrix is saved as:

```text
Visualizations/confusion_matrix.png
```

These evaluation steps are implemented in `Evaluation/evaluator.py`.

---

# 📊 Visualizations

The project generates the following visual outputs:

```text
Visualizations/
│
├── category_distribution.png
├── confusion_matrix.png
└── wordcloud.png
```

### Category Distribution

Shows the distribution of resume categories in the dataset.

### Word Cloud

Provides a visual representation of frequently occurring words in the resume text.

### Confusion Matrix

Shows the classification performance across the different resume categories.

These files are present in the project's `Visualizations` folder.

---

# 💾 Saved Model

The trained model and TF-IDF vectorizer are saved using `joblib`.

```text
Saved_Model/
│
├── resume_classifier.pkl
└── tfidf_vectorizer.pkl
```

The classifier is saved as:

```text
Saved_Model/resume_classifier.pkl
```

The TF-IDF vectorizer is saved as:

```text
Saved_Model/tfidf_vectorizer.pkl
```

This allows the trained system to be reused without retraining the model.

---

# 🔮 Resume Prediction

The project supports prediction on a new resume.

The prediction process is:

```text
┌──────────────────────┐
│   New Resume Text    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Saved TF-IDF         │
│ Vectorizer           │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Numerical Features   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Saved Logistic       │
│ Regression Model     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Predicted Category   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Confidence Score     │
└──────────────────────┘
```

The predictor loads both saved files and accepts resume text from the user. It then returns the predicted category and confidence score.

---

# 📁 Folder Structure

The actual project structure is:

```text
NLP/
│
├── Data/
│   └── Resume Screening.csv
│
├── Evaluation/
│   └── evaluator.py
│
├── Models/
│   └── resume_classifier.py
│
├── Prediction/
│   └── predictor.py
│
├── Saved_Model/
│   ├── resume_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── Terminal output/
│
├── Training/
│   └── trainer.py
│
├── Visualizations/
│   ├── Visualizer.py
│   ├── category_distribution.png
│   ├── confusion_matrix.png
│   └── wordcloud.png
│
├── preprocessing/
│   ├── __init__.py
│   ├── data_splitter.py
│   ├── dataset.py
│   ├── feature_extrator.py
│   └── text_preprocessor.py
│
├── main.py
├── requirements.txt
└── README.md
```

This structure matches the files currently present in the GitHub `NLP` directory.

---

# 🛠️ Technologies & Libraries

The project uses the following libraries:

| Library      | Purpose                                                         |
| ------------ | --------------------------------------------------------------- |
| Pandas       | Dataset handling                                                |
| NumPy        | Numerical operations                                            |
| Matplotlib   | Visualization                                                   |
| Seaborn      | Data visualization                                              |
| Scikit-learn | TF-IDF, encoding, splitting, Logistic Regression and evaluation |
| NLTK         | NLP processing support                                          |
| Joblib       | Saving/loading trained objects                                  |
| WordCloud    | Word cloud generation                                           |
| Regex        | Text cleaning                                                   |
| OpenPyXL     | Spreadsheet support                                             |
| TensorFlow   | Included in project environment                                 |
| Transformers | Included in project environment                                 |
| Datasets     | Included in project environment                                 |

The exact dependencies are listed in `requirements.txt`.

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Omajamshed/Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models.git
```

## 2. Navigate to the Repository

```bash
cd Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models
```

## 3. Navigate to the NLP Project

```bash
cd NLP
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

From inside the `NLP` folder, run:

```bash
python main.py
```

The `main.py` file executes the complete workflow:

```text
Load Dataset
      ↓
Dataset Analysis
      ↓
Text Cleaning
      ↓
Visualization
      ↓
TF-IDF Feature Extraction
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Saving
      ↓
Resume Prediction
```

The execution sequence is implemented directly in `main.py`.

---

# 📌 Main Components

### `main.py`

Controls the complete NLP pipeline and connects all project modules.

### `dataset.py`

Loads the dataset and performs basic dataset exploration such as:

* Shape
* Columns
* Dataset information
* Missing values
* Duplicate records
* First five rows
* Category distribution

### `text_preprocessor.py`

Cleans and preprocesses resume text.

### `feature_extrator.py`

Converts resume text into TF-IDF features and encodes target categories.

### `data_splitter.py`

Splits the feature matrix and labels into training and testing data.

### `resume_classifier.py`

Creates and trains the Logistic Regression classifier.

### `evaluator.py`

Evaluates the model and generates the confusion matrix.

### `trainer.py`

Saves the trained classifier and TF-IDF vectorizer.

### `predictor.py`

Loads the saved model and predicts the category of a new resume.

### `Visualizer.py`

Creates visualizations such as category distribution and word cloud.

---

# 🔄 End-to-End Pipeline

```text
                  RESUME SCREENING SYSTEM

                         START
                           │
                           ▼
              ┌─────────────────────┐
              │   Load Dataset      │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Explore Dataset     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Clean Resume Text   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │    TF-IDF           │
              │ Feature Extraction  │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Label Encoding     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Train/Test Split   │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Logistic Regression │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │    Evaluation       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │    Save Model       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ Resume Prediction   │
              └──────────┬──────────┘
                         ▼
                        END
```

---

# 🎓 Learning Outcomes

Through this project, the following concepts were practiced:

* Natural Language Processing
* Resume Text Classification
* Text Cleaning
* Regular Expressions
* TF-IDF Vectorization
* Label Encoding
* Train/Test Splitting
* Logistic Regression
* Classification Evaluation
* Confusion Matrix
* Word Cloud Visualization
* Model Persistence with Joblib
* Resume Prediction

---

# 🚀 Future Improvements

Possible improvements include:

* Use advanced NLP embeddings.
* Implement Word2Vec or GloVe.
* Replace TF-IDF with transformer embeddings.
* Experiment with LSTM/GRU models.
* Implement BERT-based resume classification.
* Add a web interface for resume uploading.
* Add automated resume ranking.
* Deploy the model as an API.
* Add batch resume processing.

---

# ⚠️ Important Note

Although this project is included in the **Deep Learning Models repository**, the current NLP implementation specifically uses **TF-IDF + Logistic Regression**, which is a traditional machine-learning NLP pipeline rather than a neural-network architecture.

This README intentionally documents the **actual implementation in the repository** rather than incorrectly describing it as an RNN, LSTM, or embedding-based Deep Learning model.

---

# 📝 Conclusion

This project demonstrates a complete **NLP-based Resume Screening and Classification system**.

Raw resume text is first cleaned and normalized. The cleaned text is then transformed into numerical features using TF-IDF. Resume categories are encoded into numerical labels, and a Logistic Regression classifier is trained on the resulting feature matrix.

The trained model is evaluated using accuracy, classification reports, and a confusion matrix. Finally, both the classifier and TF-IDF vectorizer are saved so that new resumes can be classified without retraining the system.

The project provides practical experience in building an end-to-end NLP classification pipeline from raw textual data to real-time resume category prediction.

---

# 👩‍💻 Author

**Oma Jamshed**

BS Computer Science
University of Karachi

---

⭐ **NLP Resume Screening Project — Deep Learning Models Repository**
