# 🧠 MNIST Handwritten Digit Classifier (CNN)

A Convolutional Neural Network (CNN) built using **TensorFlow + Keras** to classify handwritten digits from the popular **MNIST** dataset.  
This project achieves **99% test accuracy**, making it a strong baseline deep learning model for image classification.

---

## 📌 Project Overview

The MNIST dataset contains **70,000 grayscale images** of handwritten digits (0–9), each of size **28×28 pixels**.

This project:
- Loads and preprocesses the MNIST dataset  
- Builds a CNN architecture using TensorFlow  
- Trains the model with high accuracy  
- Evaluates performance  
- Makes predictions on test samples  
- Saves the model using the latest Keras format (`.keras`)  

---

## 📊 Model Accuracy

| Metric | Value |
|--------|--------|
| **Test Accuracy** | **99.09%** |
| **Loss** | 0.031 |

---

## 🧱 Model Architecture

```text
Conv2D (32 filters, 3×3)  
MaxPooling2D (2×2)  
Conv2D (64 filters, 3×3)  
MaxPooling2D (2×2)  
Flatten  
Dense (128 units, ReLU)  
Dense (10 units, Softmax)
