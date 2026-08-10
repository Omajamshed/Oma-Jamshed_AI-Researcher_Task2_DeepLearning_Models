# 🌸 Transfer Learning for Flower Image Classification

A Deep Learning project that uses **Transfer Learning with MobileNetV2** to classify flower images into five different categories using the **TF Flowers dataset**.

---

# 🌸 Project Overview

This project implements **Transfer Learning for Flower Image Classification** using a pretrained **MobileNetV2 Convolutional Neural Network**.

The main purpose of this project is to classify flower images into five different categories using the **TF Flowers dataset**.

Instead of training a deep CNN completely from scratch, a MobileNetV2 model pretrained on **ImageNet** is used to reuse previously learned visual features.

The project consists of two main stages:

1. **Transfer Learning / Feature Extraction**
2. **Fine-Tuning**

The final model classifies flower images into:

- Daisy
- Dandelion
- Roses
- Sunflowers
- Tulips

---

# 🎯 Objectives

The main objectives of this project are:

- Understand the concept of Transfer Learning.
- Use a pretrained CNN for image classification.
- Use MobileNetV2 with ImageNet pretrained weights.
- Load and preprocess image data.
- Resize images according to model requirements.
- Apply data augmentation.
- Build a custom classification head.
- Train the model.
- Evaluate model performance.
- Apply Fine-Tuning.
- Generate predictions on unseen images.
- Save the final trained model.

---

# 🧠 What is Transfer Learning?

Transfer Learning is a Deep Learning technique in which a model that has already learned useful features from a large dataset is reused for a new task.

Training a CNN from scratch generally requires:

- Large amounts of data
- High computational resources
- Long training time

Transfer Learning reduces these requirements by reusing knowledge learned by a pretrained model.

### General Transfer Learning Process

```text
┌──────────────────────────┐
│      Large Dataset       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Pretrained Model      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Learned Visual         │
│       Features           │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Reuse Features       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      New Dataset         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ New Classification Task  │
└──────────────────────────┘
```
## 🔄 Transfer Learning in This Project

```text
┌──────────────────────────┐
│     ImageNet Dataset     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Pretrained MobileNetV2 │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Learned Visual Features  │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    TF Flowers Dataset    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Custom Classification    │
│          Head            │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Flower Classification  │
└──────────────────────────┘

```
## 📊 Dataset
The project uses the TF Flowers dataset for flower image classification.
The dataset contains approximately 3,670 images belonging to five different flower categories.
```text
| Property          | Details              |
| ----------------- | -------------------- |
| Dataset           | TF Flowers           |
| Total Images      | ~3,670               |
| Number of Classes | 5                    |
| Image Type        | RGB                  |
| Task              | Image Classification |
| Input Size        | 224 × 224 × 3        |
```

## 🌺 Dataset Classes
The five flower classes are:
```text
┌──────────────────────────┐
│    TF Flowers Dataset    │
└────────────┬─────────────┘
             │
     ┌───────┼────────┬────────────┐
     ↓       ↓        ↓            ↓
  Daisy  Dandelion  Roses     Sunflowers
                                  │
                                  ↓
                                Tulips
```
## 🔄 Complete Project Pipeline
The complete workflow of the project is:
```text
┌───────────────────────────────┐
│       TF Flowers Dataset      │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│        Load Images            │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│      Image Preprocessing      │
│       Resize 224 × 224        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Dataset Splitting       │
│   Training / Validation/Test  │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Data Augmentation       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│         MobileNetV2           │
│      ImageNet Pretrained      │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Frozen Base Model       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Custom Classifier       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Initial Training        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│      Initial Evaluation       │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│          Fine-Tuning          │
│        Last 30 Layers         │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│       Final Evaluation        │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│          Prediction           │
└───────────────┬───────────────┘
                ↓
┌───────────────────────────────┐
│         Save Model            │
└───────────────────────────────┘
```
## 🏗️ System Architecture

The complete architecture of the Transfer Learning model is:
```text
┌─────────────────────────────┐
│        Input Image          │
│        224 × 224 × 3        │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│     Image Preprocessing     │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│        MobileNetV2          │
│                             │
│    ImageNet Pretrained      │
│     Feature Extractor       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│     Global Average          │
│        Pooling 2D           │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       Dense Layer           │
│       128 Neurons           │
│       ReLU Activation       │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│          Dropout            │
│          Rate = 0.5         │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       Output Layer          │
│        5 Neurons             │
│         Softmax             │
└──────────────┬──────────────┘
               ↓
     ┌─────────┼──────────┬────────────┐
     ↓         ↓          ↓            ↓
   Daisy   Dandelion    Roses     Sunflowers
                                          │
                                          ↓
                                        Tulips
```
## 🔁 Model State Diagram

The model passes through the following states:
```text
┌──────────────────────────┐
│     Dataset Loaded       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Image Preprocessing     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   MobileNetV2 Loaded     │
│    ImageNet Weights      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Base Model Frozen     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Classifier Training    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Initial Evaluation     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Last 30 Layers Unfrozen  │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      Fine-Tuning         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Final Evaluation       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│       Prediction         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Final Model Saved     │
└──────────────────────────┘
```

## 🖼️ Image Preprocessing
The original flower images may have different dimensions.
Therefore, all images are converted into a standard format before being passed to MobileNetV2.
Preprocessing Flow
```text
┌──────────────────────────┐
│   Original Flower Image  │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      Resize Image        │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      224 × 224 × 3       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ MobileNetV2 Preprocessing│
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Processed Image      │
└──────────────────────────┘
```


## 🔀 Dataset Splitting

The dataset is divided into training, validation, and testing subsets.
```text

                 TF Flowers Dataset
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      Training       Validation        Test
        80%             10%             10%
```

Training Dataset

Used for learning model parameters.

Validation Dataset

Used to monitor model performance during training.

Test Dataset

Used for final evaluation on unseen images.

## 🔄 Data Augmentation

Data augmentation increases the variation of training images and helps reduce overfitting.

The project uses techniques such as:

Random Horizontal Flip
Random Rotation
Random Zoom
Random Contrast
Augmentation Flow
```text
┌──────────────────────────┐
│     Original Image       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Random Horizontal Flip   │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Random Rotation      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│       Random Zoom        │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Random Contrast      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Augmented Image      │
└──────────────────────────┘
```

## 🧠 Model Architecture

The model consists of two major components:

1. Pretrained MobileNetV2

MobileNetV2 is used as the pretrained feature extractor.

The original ImageNet classification layer is removed because this project has five flower classes.

2. Custom Classification Head
```text
MobileNetV2
     ↓
GlobalAveragePooling2D
     ↓
Dense(128)
     ↓
ReLU
     ↓
Dropout(0.5)
     ↓
Dense(5)
     ↓
Softmax
     ↓
Flower Class
```

## 🔍 Feature Extraction

During the first training stage, the MobileNetV2 base model is frozen
```text
┌──────────────────────────┐
│       Input Image        │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│       MobileNetV2        │
│      Frozen Model ❄️     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Feature Extraction     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Custom Classification    │
│          Head            │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│        Training          │
└──────────────────────────┘
```
🔥 Fine-Tuning

After initial Transfer Learning, Fine-Tuning is performed.

The last 30 layers of MobileNetV2 are made trainable.

```text
             MobileNetV2
                  │
       ┌──────────┴──────────┐
       ↓                     ↓
 Early & Middle Layers   Last 30 Layers
       ↓                     ↓
    Frozen ❄️             Trainable 🔥
```

🏋️ Training

The project uses two major training stages.
```text
Stage 1 — Transfer Learning
MobileNetV2
     ↓
Base Model Frozen
     ↓
Custom Classification Head
     ↓
Initial Training
     ↓
Initial Evaluation
Stage 2 — Fine-Tuning
Last 30 Layers
     ↓
Unfreeze
     ↓
Small Learning Rate
     ↓
Fine-Tuning
     ↓
Final Model
     ↓
Final Evaluation
```
## ⚙️ Model Configuration
Parameter	Value
Base Model	MobileNetV2
Pretrained Weights	ImageNet
Input Shape	224 × 224 × 3
Number of Classes	5
Dense Layer	128 Neurons
Dense Activation	ReLU
Dropout	0.5
Output Layer	5 Neurons
Output Activation	Softmax
Optimizer	Adam
Initial Learning Rate	0.0001
Fine-Tuning Learning Rate	0.00001
Initial Epochs	10
Fine-Tuning Epochs	5
Loss Function	Sparse Categorical Crossentropy

## 📊 Evaluation

The final model is evaluated on unseen test images.

The evaluation includes:

Test Accuracy
Test Loss
Precision
Recall
F1-Score
Classification Report
Confusion Matrix


## 🔲 Confusion Matrix

The confusion matrix compares actual classes with predicted classes.
```text
                         Predicted
              ┌─────────────────────────────┐
              │ Daisy │ Dandelion │ Roses │ …│
──────────────┼───────┼───────────┼───────┼──┤
Actual Daisy  │   ✓   │           │       │  │
Dandelion     │       │     ✓     │       │  │
Roses         │       │           │   ✓   │  │
Sunflowers    │       │           │       │ ✓│
Tulips        │       │           │       │ ✓│
              └─────────────────────────────┘
```
The diagonal values represent correct predictions.

The off-diagonal values represent incorrect predictions.

## 📉 Training Graphs

The project generates training and validation graphs.

# Accuracy Graph
The accuracy graph shows how training and validation accuracy change over epochs.
# Loss Graph
The loss graph shows how training and validation loss change over epochs.

# Typical output files:
accuracy.png
loss.png

## 🌷 Prediction
After training, the final model predicts the class of an unseen flower image.
```text
┌──────────────────────────┐
│    Input Flower Image    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Resize 224 × 224      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│       MobileNetV2        │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Feature Extraction    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Classification Head     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Softmax Probabilities   │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Highest Probability      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Predicted Flower Class   │
└──────────────────────────┘
```
Possible predictions:
Daisy
Dandelion
Roses
Sunflowers
Tulips

## 🖼️ Actual vs Predicted

The project can visualize actual and predicted labels.

Correct Prediction
Actual: Roses
Predicted: Roses

✓ Correct
Incorrect Prediction
Actual: Tulips
Predicted: Roses

✗ Incorrect
📤 Output Files

The project generates outputs such as:
```text
Outputs/
│
├── accuracy.png
├── loss.png
├── confusion_matrix.png
├── final_confusion_matrix.png
├── prediction.png
└── MobileNetV2_Flower_TransferLearning.keras
```
File	Purpose
accuracy.png	Training and validation accuracy graph
loss.png	Training and validation loss graph
confusion_matrix.png	Initial confusion matrix
final_confusion_matrix.png	Final confusion matrix
prediction.png	Prediction visualization
.keras	Saved final model
```text
📁 Project Folder Structure
TRANSFER_LEARNING_MODEL/
│
├── Data/
│
├── Outputs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── final_confusion_matrix.png
│   ├── prediction.png
│   └── MobileNetV2_Flower_TransferLearning.keras
│
├── TL_Preprocessing.py
├── TL_Model.py
├── TL_Training.py
├── TL_Evaluation.py
├── TL_Prediction.py
├── TL_Main.py
│
└── README.md
```
## 📂 File Descriptions
File / Folder	Purpose
Data/	Dataset-related files
Outputs/	Graphs, predictions and saved model
TL_Preprocessing.py	Dataset loading and preprocessing
TL_Model.py	MobileNetV2 model architecture
TL_Training.py	Model training
TL_Evaluation.py	Model evaluation
TL_Prediction.py	Prediction and visualization
TL_Main.py	Main project execution
README.md	Complete project documentation
## 🛠️ Technologies Used
Python
TensorFlow
Keras
MobileNetV2
NumPy
Matplotlib
Seaborn
Scikit-learn
Google Colab

## 📦 Installation

For Google Colab, most required libraries are already available.

If required:

pip install tensorflow
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn

For a local environment:

python -m venv venv

Windows:

venv\Scripts\activate

Install dependencies:

pip install tensorflow numpy matplotlib seaborn scikit-learn

## ▶️ How to Run

Run the project in the following order:
```text
1. Load Dataset
       ↓
2. Preprocess Images
       ↓
3. Split Dataset
       ↓
4. Apply Data Augmentation
       ↓
5. Load MobileNetV2
       ↓
6. Build Classification Head
       ↓
7. Train Initial Model
       ↓
8. Evaluate Initial Model
       ↓
9. Unfreeze Last 30 Layers
       ↓
10. Fine-Tune Model
       ↓
11. Evaluate Final Model
       ↓
12. Generate Predictions
       ↓
13. Save Final Model
```
## 💻 Google Colab GPU

For faster training, a GPU can be enabled in Google Colab.
```text
Go to:

Runtime
   ↓
Change Runtime Type
   ↓
Hardware Accelerator
   ↓
T4 GPU

Check GPU availability:

import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
```
## 💾 Loading the Saved Model

The saved model can be loaded without retraining:

from tensorflow.keras.models import load_model

model = load_model(
    "MobileNetV2_Flower_TransferLearning.keras"
)

print("Model loaded successfully!")
## 📊 Results

After final evaluation, record the actual model performance.

Final Test Accuracy: [0.000]%
Final Test Loss: [0.00]

The classification report provides:

Precision
Recall
F1-Score
Support

for each flower class.

## 🔥 Initial Training vs Fine-Tuning
```text
┌──────────────────────────┐
│     Transfer Learning    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Frozen MobileNetV2      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Train Classifier       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Initial Evaluation     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      Fine-Tuning         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Unfreeze Last 30 Layers  │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Small Learning Rate    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│     Final Training       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Final Evaluation      │
└──────────────────────────┘
```
## ⚖️ Feature Extraction vs Fine-Tuning
Feature Extraction	Fine-Tuning
Base model frozen	Selected layers unfrozen
Faster training	Slower training
Only classifier trained	Some pretrained layers trained
Reuses learned features	Adapts learned features
First stage	Second stage
## ⚖️ Transfer Learning vs Training From Scratch
Feature	Training From Scratch	Transfer Learning
Initial Weights	Random	ImageNet
Training Time	Higher	Lower
Data Requirement	Higher	Lower
Feature Learning	From beginning	Reused
Pretrained Knowledge	❌	✅
Fine-Tuning	❌	✅
Computational Cost	Higher	Lower
✅ Advantages
Uses pretrained ImageNet knowledge.
Reduces training time.
Requires less data than training from scratch.
MobileNetV2 is lightweight.
Provides efficient feature extraction.
Data augmentation improves generalization.
Fine-Tuning provides task-specific adaptation.
Supports multiple evaluation metrics.
Final model can be saved and reused.
## ⚠️ Limitations
Only five flower categories are included.
Some flower categories have visually similar characteristics.
Model performance depends on image quality.
Fine-Tuning requires additional computational resources.
The model may perform differently on external images.
Dataset size is relatively small compared with large-scale image datasets.
🚀 Future Improvements

Future improvements can include:

Using a larger flower dataset.
Adding more flower categories.
Experimenting with EfficientNet.
Experimenting with ResNet50.
Experimenting with DenseNet.
Experimenting with InceptionV3.
Applying Learning Rate Scheduling.
Applying Early Stopping.
Performing Hyperparameter Optimization.
Deploying the model using Streamlit.
Creating a web-based flower classification application.
Adding real-time camera-based classification.
Testing the model using external real-world images.
## 📚 Learning Outcomes

Through this project, the following concepts were practiced:

Deep Learning
Convolutional Neural Networks
Transfer Learning
MobileNetV2
ImageNet
Image Classification
Image Preprocessing
Data Augmentation
Feature Extraction
Fine-Tuning
Model Training
Model Evaluation
Classification Reports
Confusion Matrices
Prediction Visualization
TensorFlow
Keras
Google Colab
## 🧠 Complete Concept Summary

```text
┌──────────────────────────┐
│    TF Flowers Dataset    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Image Preprocessing    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Data Augmentation     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│  Pretrained MobileNetV2  │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Feature Extraction     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Custom Classification    │
│          Head            │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│    Initial Training      │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│       Evaluation         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      Fine-Tuning         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Final Evaluation       │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│       Prediction         │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│      Model Saving        │
└──────────────────────────┘
```
🎓 Conclusion

This project demonstrates a complete Transfer Learning pipeline for flower image classification using a pretrained MobileNetV2 model.

The TF Flowers dataset is first loaded and preprocessed. Images are resized to 224 × 224 × 3 and divided into training, validation, and testing subsets.

Data augmentation is applied to training images to improve generalization and reduce overfitting.

During the first stage, MobileNetV2 is used as a frozen feature extractor. A custom classification head is added to classify images into five flower categories.

After initial training, Fine-Tuning is performed by unfreezing the last 30 layers of MobileNetV2 and using a smaller learning rate.

The final model is evaluated using accuracy, loss, precision, recall, F1-score, classification report, confusion matrix, and prediction visualization.

The trained model is saved in .keras format and can be reused for future flower image predictions.

Overall, this project demonstrates how Transfer Learning can make image classification more efficient by reusing knowledge learned from a large pretrained dataset.

## ⭐ Project Highlights
🌸 TF Flowers Dataset
        ↓
🖼️ Image Preprocessing
        ↓
🔄 Data Augmentation
        ↓
🧠 MobileNetV2
        ↓
📚 ImageNet Pretrained Weights
        ↓
🔍 Feature Extraction
        ↓
🎯 Custom Classification Head
        ↓
🏋️ Initial Training
        ↓
🔥 Fine-Tuning
        ↓
📊 Evaluation
        ↓
🔲 Confusion Matrix
        ↓
🌷 Flower Prediction
        ↓
💾 Saved Keras Model

## 📌 Final Project Summary
Category	Details
Project Type	Deep Learning
Task	Image Classification
Dataset	TF Flowers
Total Images	~3,670
Number of Classes	5
Model	MobileNetV2
Pretrained On	ImageNet
Input Size	224 × 224 × 3
Transfer Learning	Yes
Feature Extraction	Yes
Fine-Tuning	Yes
Data Augmentation	Yes
Evaluation	Yes
Prediction	Yes
Model Format	.keras
Framework	TensorFlow / Keras
Platform	Google Colab
👩‍💻 Author

Oma Jamshed

Deep Learning & AI Projects

⭐ Project Status

Completed — Transfer Learning Flower Image Classification
