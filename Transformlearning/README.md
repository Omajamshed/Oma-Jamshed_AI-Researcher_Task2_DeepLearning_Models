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
