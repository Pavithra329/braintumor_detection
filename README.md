# 🧠 Brain Tumor Detection using Deep Learning (VGG16)

## 📍 About the Project

This project implements an automated brain tumor detection system using MRI image data and deep learning techniques. By applying **Transfer Learning** with the **VGG16** model, the system classifies brain scans into tumor categories with improved accuracy, reduced training time, and enhanced feature extraction.
This repository includes dataset handling, model training, prediction functionality, and performance evaluation scripts.

---

## ⭐ Features

✅ Classifies MRI images into tumor categories
✅ Uses Transfer Learning with VGG16
✅ Data augmentation for better generalization
✅ Model training and evaluation included
✅ Saved model for reuse 
✅ Simple prediction workflow for new images

---

## 🧬 Tumor Classes (Dataset Dependent)

* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **(Optional)** No Tumor

---

## 📂 Repository Structure

```
braintumor_detection/
├── dataset/
│   ├── Training/
│   └── Testing/
├── model/
│   └── tumor_detection_model.h5
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── results/
├── README.md
└── requirements.txt

## 🛠 Tech Stack

| Component   | Technology                        |
| ----------- | --------------------------------- |
| Language    | Python                            |
| Framework   | TensorFlow / Keras                |
| Model       | VGG16                             |
| Libraries   | NumPy, Pillow, Scikit-learn       |
| Environment | Jupyter Notebook / Python Scripts |

---

## 🚀 How to Run the Project

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/Pavithra329/braintumor_detection
cd braintumor_detection
```

### ✅ 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 3. Train the Model

```bash
python src/train.py
```

### ✅ 4. Predict on a New MRI Image

```bash
python src/predict.py --image sample.jpg
```

---

## 🔁 Workflow

1. Load dataset
2. Preprocess MRI images
3. Apply augmentation
4. Load VGG16 without top layers
5. Add custom classifier
6. Train the model
7. Evaluate performance
8. Predict new images

---

## ✅ Conclusion

The project confirms that deep learning models such as VGG16 can efficiently classify brain tumors from MRI scans with high accuracy, supporting early diagnosis and clinical decision-making.

## 🙏 Acknowledgment

This project is developed for academic and research learning in medical imaging and artificial intelligence.

---

## 📜 License

This project is intended for **educational and research purposes only** and not for clinical use.

---

