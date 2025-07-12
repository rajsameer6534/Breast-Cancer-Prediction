# 🧠 Breast Cancer Prediction with ML | Flask App

A machine learning-based diagnostic system for early detection of breast cancer using clinical data. This project classifies tumors as benign or malignant with 95%+ accuracy, and includes a web-based prediction interface built using Flask.

![Breast Cancer Prediction Screenshot]


---![FotoJet](https://github.com/user-attachments/assets/9989ffc4-b4a8-4e69-b0b3-c6441316572f)


## 🚀 Project Overview

Early detection of breast cancer can significantly increase survival rates. This project leverages machine learning to analyze clinical features and predict the likelihood of cancer. With a user-friendly Flask interface, it simulates real-time usage by healthcare practitioners.

---

## 📊 Problem Statement

> Predict whether a tumor is **benign** or **malignant** based on key diagnostic features extracted from clinical data (e.g., mean radius, texture, concavity, symmetry, etc.).

The goal is to build a robust classification model that:
- Generalizes well on unseen patient data
- Maintains high interpretability
- Supports real-time usage via a lightweight web interface

---

## 🧠 Technologies & Tools Used

| Category              | Tools/Technologies                        |
|----------------------|--------------------------------------------|
| Programming Language | Python                                     |
| Data Processing      | Pandas, NumPy                              |
| Modeling             | scikit-learn (Logistic Regression, SVM, Random Forest) |
| Evaluation Metrics   | Accuracy, ROC-AUC, Confusion Matrix        |
| Deployment           | Flask (Web Framework)                      |
| Visualization        | Seaborn, Matplotlib                        |
| Dev Tools            | Git, Jupyter Notebook                      |

---

## 🛠️ Features

- ✅ Data cleaning & preprocessing pipeline (scaling, outlier removal)
- ✅ Multiple ML algorithms tested & compared
- ✅ ROC-AUC and cross-validation used for evaluation
- ✅ Flask app with interactive UI for real-time predictions
- ✅ Lightweight, modular, and easy to deploy

---

## 📁 Project Structure

```bash
breast-cancer-prediction/
│
├── static/                 # CSS files (if applicable)
├── templates/              # HTML templates for Flask
│   └── index.html
├── model/                  # Serialized ML model
│   └── model.pkl
├── app.py                  # Flask application
├── data.csv                # Dataset used (optional or linked)
├── preprocess.ipynb        # Data cleaning and EDA
├── model_training.ipynb    # Model development and evaluation
├── requirements.txt        # Python dependencies
└── README.md               # You're here!
