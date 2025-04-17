# 🧠 Brain CT Scan Classification with CNN

This project uses a Convolutional Neural Network (CNN) to classify brain CT scans as either **normal** or **hemorrhage**. It's a practical deep learning application for medical image classification, implemented using TensorFlow and Keras.

---

## 🧾 Project Overview

### 🎯 Objective
We aim to train a deep learning model that can accurately detect hemorrhages in CT scan images of the brain. This is critical in medical diagnostics, where rapid identification can be life-saving.
---

## 🧠 Background

### Why Deep Learning?
Deep Learning, particularly Convolutional Neural Networks (CNNs), excels at tasks involving **image recognition**. In medical imaging, CNNs can learn complex visual patterns to assist radiologists by automating the detection process.

### Dataset
- 140 training images: 70 normal, 70 hemorrhage
- 60 validation images: 30 normal, 30 hemorrhage
- 10 test images for final evaluation
- Organized in directory format for use with Keras' `ImageDataGenerator`

Folder structure:
data/ ├── head_ct_slices/ │ ├── train/ │ │ ├── normal/ │ │ └── hemorrhage/ │ ├── validate/ │ │ ├── normal/ │ │ └── hemorrhage/ │ └── test/

## 🧬 Model Architecture

We built a deep CNN using the following structure:
Input (150x150x3)
→ Conv2D(16) + MaxPool
→ Conv2D(32) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(64) + MaxPool
→ Conv2D(128) + MaxPool
→ Flatten
→ Dense(256, ReLU)
→ Dense(1, Sigmoid)
Each convolution layer extracts features. Pooling layers reduce spatial size and computation. Final layers interpret these features to classify the image.

## 📊 Training Performance
Epochs: 10
Optimizer: Adam
Loss: Binary Crossentropy
Metric: Accuracy

Training output includes:
- `accuracy` and `val_accuracy` for performance tracking
- `loss` and `val_loss` to monitor potential overfitting

---

## 🔍 Evaluation

After training, the model is tested on **unseen test images**. It predicts each image as either `"normal"` or `"hemorrhage"` based on a probability threshold of **0.5**:

```python
if prediction < 0.5:
    print("Hemorrhage")
else:
    print("Normal")
```
---

## 📌 Key Learnings

- How to use `ImageDataGenerator` for medical images
- Building and tuning CNN layers for feature extraction
- Interpreting model predictions
- Importance of validation and test sets in medical AI

---

## 🧠 Author Notes

This project is designed as both a **practical demo** and a **learning tool**.  
All steps and code are explained in plain language in the script file so that learners can follow along.

