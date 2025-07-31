# 🍅 Tomato Disease Detection Using Machine Learning Algorithms

Welcome to the official repository of **Tomato Disease Detection Using Machine Learning Algorithms**, a comparative study that explores how deep learning architectures like **InceptionV3** and **MobileNetV2** can be used to detect and classify common tomato plant diseases from leaf images with high accuracy.

---

## 📌 Project Overview

Tomatoes are one of the most consumed vegetables globally, and their cultivation is often impacted by diseases that cause significant yield loss. Early and accurate identification of these diseases is essential. This project presents a machine learning-based solution to classify tomato leaf diseases using image classification models.

Trained and evaluated:
- 🧠 **InceptionV3**: Achieved 97% accuracy
- ⚡ **MobileNetV2**: Achieved 95% accuracy

---

## 🦠 Diseases Detected
The system identifies the following tomato leaf conditions:
- **Early Blight**
- **Late Blight**
- **Bacterial Rot**
- **Two-Spotted Spider Mite**
- **Healthy Leaf**

---

## 🧪 Dataset

- 📸 **Total Images**: 6143 (Training), 500 (Validation)
- 🗂️ **Classes**: 5 (4 Diseased + 1 Healthy)
- 🧹 **Preprocessing**: Image resizing, normalization, data augmentation

> ⚠️ Due to size constraints, only a **sample** of the dataset is included.

---

## 🔍 Model Architectures

- **InceptionV3**: Known for its depth and multi-scale feature extraction.
- **MobileNetV2**: Lightweight model ideal for mobile and embedded systems.

> Both models were fine-tuned with transfer learning using TensorFlow/Keras.

---

## 📊 Results

| Model        | Accuracy | Parameters | Inference Time |
|--------------|----------|------------|----------------|
| InceptionV3  | 97%      | 23.8M      | Medium         |
| MobileNetV2  | 95%      | 3.5M       | Fast           |

- ✅ The output images available in the [`outputs/`](outputs/) folder.

