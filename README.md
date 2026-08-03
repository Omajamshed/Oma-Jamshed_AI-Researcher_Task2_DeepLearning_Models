
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
Ek aur suggestion
Agar tum GitHub profile ko portfolio level banana chahti ho, to main is README ko aur attractive bana sakta hoon jisme:

🛡️ Technology badges (Python, TensorFlow, Keras, Scikit-learn)

📊 Accuracy & Loss graphs automatically display hon

🖼️ Project architecture image

🎯 Features section

📈 Results table

📸 Screenshots

Ye README dekhte hi professional impression deta hai.


ab  may chati hu k is repo k andar ak akmodelkam folder ho or us folder k andar saray modelski file and folder ho jis tarha deep laeraning rep ka nam h i want k is repo k anda ak annmodel ka folder ho or us k andar saray annmodel  project say releated files and folderho pir cnn and same goes for oothers

Ye structure zyada professional lagega. Main bhi isi tarah organize karta. 👍

Tumhari repository ka naam rahega:

Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models
Aur uske andar har model ka apna folder hoga.

Recommended Folder Structure
Oma-Jamshed_AI-Researcher_Task2_DeepLearning_Models/
│
├── README.md
├── requirements.txt
│
├── ANN_Model/
│   ├── Data/
│   ├── Preprocessing/
│   ├── Models/
│   ├── Training/
│   ├── Evaluation/
│   ├── Prediction/
│   ├── Graphs/
│   ├── best_ann_model.keras
│   ├── main.py
│   └── README.md
│
├── CNN_Model/
│   ├── Data/
│   ├── Preprocessing/
│   ├── Models/
│   ├── Training/
│   ├── Evaluation/
│   ├── Prediction/
│   ├── Graphs/
│   ├── main.py
│   └── README.md
│
├── RNN_Model/
│   ├── Data/
│   ├── Preprocessing/
│   ├── Models/
│   ├── Training/
│   ├── Evaluation/
│   ├── Prediction/
│   ├── Graphs/
│   ├── main.py
│   └── README.md
│
├── LSTM_Model/
│   ├── Data/
│   ├── Preprocessing/
│   ├── Models/
│   ├── Training/
│   ├── Evaluation/
│   ├── Prediction/
│   ├── Graphs/
│   ├── main.py
│   └── README.md
│
└── Transfer_Learning/
    ├── Data/
    ├── Preprocessing/
    ├── Models/
    ├── Training/
    ├── Evaluation/
    ├── Prediction/
    ├── Graphs/
    ├── main.py
    └── README.md
Abhi ANN ko move karna hai
Terminal mein:

mkdir ANN_Model
Ab ANN project ki sari files is folder ke andar move kar do.

Final structure ANN ka aisa hona chahiye:

ANN_Model/
│
├── Data/
├── Preprocessing/
├── Models/
├── Training/
├── Evaluation/
├── Prediction/
├── Graphs/
├── best_ann_model.keras
├── main.py
├── README.md
└── requirements.txt
Uske baad GitHub update karne ke liye
git add .
git commit -m "Restructured repository into separate deep learning model folders"
git push
Repository ka Main README
Root README.md sirf overview dega:

# Deep Learning Models

This repository contains multiple deep learning implementations developed as part of an AI Researcher task.

## Models Included

- ✅ Artificial Neural Network (ANN)
- 🔄 Convolutional Neural Network (CNN)
- 🔄 Recurrent Neural Network (RNN)
- 🔄 Long Short-Term Memory (LSTM)
- 🔄 Transfer Learning

## Repository Structure

- ANN_Model/
- CNN_Model/
- RNN_Model/
- LSTM_Model/
- Transfer_Learning/
💡 Ye structure GitHub par bahut professional lagega, aur jab CNN, RNN, LSTM aur Transfer Learning complete karogi to bas unke folders add karti jaogi. Is tarah ek hi repository mein tumhare saare deep learning models neatly organized rahenge.


d81d34c5-6618-4857-96b2-16e70293bc53.png
 yeh banawa to h folder

Perfect! 🎉 Ab Ann Model folder ban gaya hai.

Lekin abhi bhi sari files uske bahar hain. Chalo terminal se hi move kar dete hain. PowerShell mein ye commands chalao.

1. Sab folders move karo
Move-Item Data "Ann Model"
Move-Item Evaluation "Ann Model"
Move-Item Models "Ann Model"
Move-Item Prediction "Ann Model"
Move-Item Preprocessing "Ann Model"
Move-Item Training "Ann Model"
Move-Item Visualizations "Ann Model"
Move-Item "terminal output" "Ann Model"
2. Sab files move karo
Move-Item main.py "Ann Model"
Move-Item requirements.txt "Ann Model"
Move-Item accuracy.png "Ann Model"
Move-Item loss.png "Ann Model"
Move-Item best_ann_model.keras "Ann Model"
⚠️ .venv aur .idea ko move mat karna. Ye root folder mein hi rehne chahiye.

3. Check karo
dir
Ab output kuch aisa hona chahiye:

.idea
.venv
Ann Model
Aur agar ye chalao:

dir "Ann Model"
To uske andar tumhari sari folders aur files dikhni chahiye.

4. GitHub update
git add .
git commit -m "Reorganized project into Ann Model folder"
git push
💡 Meri recommendation
Folder ka naam Ann Model ki jagah ANN_Model rakhna zyada professional hai.

Rename karne ke liye:

Rename-Item "Ann Model" "ANN_Model"
Iske baad final structure hoga:

DeepLearning_Models/
│
├── .venv/
├── .idea/
├── ANN_Model/
│   ├── Data/
│   ├── Preprocessing/
│   ├── Models/
│   ├── Training/
│   ├── Evaluation/
│   ├── Prediction/
│   ├── Visualizations/
│   ├── main.py
│   ├── requirements.txt
│   ├── accuracy.png
│   ├── loss.png
│   └── best_ann_model.keras
Ye structure GitHub aur future CNN/RNN/LSTM folders ke liye bilkul professional lagega.


jo move krnay k sari files and folder wo ak he bari m lik do may bus yaha say copy k r k waha ppaste krdu gi 

Haan, bilkul. Ye ek hi command block copy karke PowerShell mein paste kar do.

Move-Item Data "Ann Model"
Move-Item Evaluation "Ann Model"
Move-Item Models "Ann Model"
Move-Item Prediction "Ann Model"
Move-Item Preprocessing "Ann Model"
Move-Item Training "Ann Model"
Move-Item Visualizations "Ann Model"
Move-Item "terminal output" "Ann Model"
Move-Item main.py "Ann Model"
Move-Item requirements.txt "Ann Model"
Move-Item accuracy.png "Ann Model"
Move-Item loss.png "Ann Model"
Move-Item best_ann_model.keras "Ann Model"
⚠️ In files/folders ko move mat karna:

.venv

.idea

.git

.gitignore (agar hai)

Ye root folder mein hi rehne chahiye.

Uske baad GitHub update ke liye ye commands chala dena:

git add .
git commit -m "Reorganized ANN project into ANN_Model folder"
git push
Ek suggestion: Folder ka naam Ann Model ki jagah ANN_Model kar lo. Ye naming convention zyada professional lagti hai aur baad mein CNN_Model, RNN_Model, LSTM_Model ke saath consistent bhi rahegi.


folder k andar wali sari files may red color arah h 

Ghabrana mat. ❤️ Ye normal hai.

PyCharm mein red color ka matlab hota hai Git ne files ko "untracked" ya "moved" detect kiya hai. Ye error nahi hota.

Pehle check karo
Terminal mein ye command chalao:

git status
Aur uska output yahan paste karo.

Agar output kuch aisa aaye:
Changes not staged for commit:
deleted: Data/...
new file: Ann Model/Data/...
ya

Untracked files:
Ann Model/
To iska matlab sab theek hai. Bas Git ko changes save karne hain.

Commands:

git add .
git commit -m "Reorganized ANN project into ANN_Model folder"
git push
Agar PyCharm mein red color hi dikh raha ho
To:

VCS → Git → Refresh File Status

Ya PyCharm restart kar do.

Mujhe sirf git status ka output bhej do.
Usse main 100% confirm kar dunga ke sab theek hai ya koi aur issue hai.


Chats with attachments paused
You’ve used all your messages with Instant until 2:37 PM. Upgrade to keep working with files and images here, or start a new chat to continue with a lower quality model.

New chat

Claim free offer

