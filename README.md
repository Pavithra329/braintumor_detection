# 🧠 Brain Tumor Classification using VGG16 (Transfer Learning)

---

## 1. Project Introduction

This project is a Deep Learning implementation focused on classifying **Brain Tumors** from MRI images. Utilizing **Transfer Learning** with the **VGG16** architecture, it aims to differentiate between different tumor types (e.g., Glioma, Meningioma, Pituitary) and healthy scans. Developed purely as a learning exercise, it demonstrates core concepts in Convolutional Neural Networks (CNNs) and image augmentation.

## 2. Project Overview

The core of this project involves fine-tuning a pre-trained VGG16 model on a dataset of brain MRI images. The process includes:

* **Data Loading and Preprocessing:** Reading images from directories and preparing them for the model.
* **Image Augmentation:** Applying random brightness and contrast changes to the training data *on-the-fly* using a custom data generator to improve generalization and prevent overfitting.
* **Model Building:** Freezing the majority of the VGG16 layers and training the final dense layers for multi-class classification using **sparse categorical crossentropy**.
* **Training and Evaluation:** Training the model using batches and saving the final model in the robust TensorFlow **SavedModel** format for future use.

## 3. Tech Stack

| Category | Component | Purpose |
| :--- | :--- | :--- |
| **Deep Learning Framework** | `TensorFlow` / `Keras` | Core framework for building and training the CNN model. |
| **Model Architecture** | `VGG16` (Pre-trained) | Base model for transfer learning. |
| **Data Handling** | `NumPy`, `os`, `PIL` (Pillow) | Efficient array operations and image manipulation. |
| **Utilities** | `scikit-learn` (`shuffle`) | Data randomization and utility functions. |

## 4. Project Features

✨ **Key Features of the Classifier:**

* **Transfer Learning Implementation:** Efficiently leverages pre-trained weights from the massive ImageNet dataset using VGG16.
* **Custom Data Generation:** Uses a `datagen` function for batch loading and real-time image augmentation, minimizing memory usage.
* **On-the-fly Augmentation:** Applies random **brightness** and **contrast** shifts to images during training, making the model more robust.
* **Model Persistence:** Efficiently saves the final trained model using the **TensorFlow SavedModel** format for easy loading and deployment.
* **Multi-Class Classification:** Capable of classifying between multiple tumor types and non-tumor cases based on the provided dataset.

## 5. Project Demo

Since this is a command-line deep learning project, a typical "demo" involves running the prediction script.

To demonstrate the classifier's functionality:

1.  A pre-trained model (`my_brain_tumor_classifier` directory) is loaded.
2.  A new, unseen MRI image is passed to the **`preprocess_image`** function.
3.  The model runs **`loaded_model.predict(image)`** and outputs a probability score for each class.

> **Example Output:** The image is classified as **Pituitary Tumor** with a confidence score of **94.5%**.

## 6. Project Installation and Setup

Follow these steps to set up and run the project locally.

### Prerequisites

You need **Python 3.7+** installed.

```bash
# Recommended: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate   # On Windows
```