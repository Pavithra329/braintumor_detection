# 🧠 Brain Tumor Classification using Deep Learning

## 📘 Project Introduction
This project focuses on developing a deep learning-based system to classify MRI brain images into different tumor types.  
It leverages a pre-trained VGG16 convolutional neural network to enhance accuracy and performance.  
This work is built solely for **learning and research purposes** in computer vision and medical image analysis.

---

## 🔍 Project Overview
Brain tumor detection plays a vital role in medical diagnostics, where early and accurate classification can help improve treatment outcomes.  
In this project, a **Convolutional Neural Network (CNN)** model is built using **Transfer Learning** with **VGG16**, trained on MRI images of the brain.  
The model is fine-tuned to differentiate between various tumor classes (e.g., glioma, meningioma, pituitary) and healthy scans.  
It performs image preprocessing, augmentation, training, and evaluation, and outputs classification metrics such as accuracy and classification reports.

---

## 🧰 Tech Stack Used
- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow / Keras  
- **Pre-trained Model:** VGG16  
- **Libraries:** NumPy, PIL (Pillow), scikit-learn  
- **Development Environment:** Jupyter Notebook / Python Script  
- **Dataset:** MRI Brain Images (Training & Testing folders)

---

## 🌟 Project Features
- Image **augmentation** for improving model generalization.  
- **Transfer Learning** using the pre-trained **VGG16** network.  
- Custom **fully connected layers** for classification.  
- **Model saving** and reusability for future predictions.  
- **Evaluation metrics** including classification report and accuracy.  
- Clean and modular **code structure** for easy understanding and modification.

---

## 🎥 Project Demo
1. Load and preprocess MRI images.  
2. Train the model using the provided dataset.  
3. Evaluate performance and generate classification reports.  
4. Save the trained model (`my_brain_tumor_classifier/`) for deployment or testing.  

You can visualize training results using TensorFlow’s built-in history object or integrate with tools like **TensorBoard** for advanced monitoring.

---

## ⚙️ Installation and Setup

### Prerequisites
Ensure you have Python 3.8+ installed and the following dependencies:

```bash
pip install tensorflow numpy pillow scikit-learn
