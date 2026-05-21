# 📩 Spam SMS Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## 📌 Project Overview

This project builds a machine learning model to classify SMS messages as **Spam** or **Legitimate (Ham)**. It uses Natural Language Processing (NLP) techniques with TF-IDF vectorization and compares two classification algorithms: **Naive Bayes** and **Logistic Regression**.

---

## 🎯 Objectives

- Build a model to classify SMS messages as spam or legitimate
- Convert text into numerical form using TF-IDF
- Apply and compare Naive Bayes and Logistic Regression algorithms
- Evaluate model accuracy and performance metrics

---

## 📂 Project Structure

```
spam_detection/
│
├── spam.csv                  # Dataset (SMS Spam Collection)
├── spam_detection.ipynb      # Main Jupyter Notebook
├── spam_distribution.png     # Spam vs Ham bar chart
├── message_length.png        # Message length distribution
├── confusion_matrix.png      # Model confusion matrices
└── README.md                 # Project documentation
```

---

## 📊 Dataset

- **Source:** [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 SMS messages
- **Classes:** Ham (Legitimate) and Spam
- **Format:** CSV with label and message columns

---

## 🛠️ Technologies Used

| Library | Purpose |
|--------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `nltk` | Text preprocessing (stopwords, stemming) |
| `scikit-learn` | TF-IDF, ML models, evaluation |
| `matplotlib` | Data visualization |
| `seaborn` | Confusion matrix heatmap |

---

## ⚙️ Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

---

## 🔄 Workflow

```
Raw SMS Data
    ↓
Text Preprocessing (lowercase, remove special chars, stemming)
    ↓
TF-IDF Vectorization (5000 features, bigrams)
    ↓
Train/Test Split (80% / 20%)
    ↓
Model Training (Naive Bayes + Logistic Regression)
    ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
```

---

## 🧹 Text Preprocessing Steps

1. Convert to lowercase
2. Remove special characters and numbers
3. Tokenization
4. Remove stopwords
5. Apply Porter Stemming

---

## 🤖 Models Used

### 1. Multinomial Naive Bayes
- Best suited for text classification with TF-IDF
- Fast training and prediction
- Works well with sparse matrices

### 2. Logistic Regression
- Strong baseline for binary classification
- Provides probability scores
- Handles high-dimensional text features well

---

## 📈 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Naive Bayes | ~97% | ~0.97 |
| Logistic Regression | ~98% | ~0.99 |

---

## 🧪 Sample Predictions

```python
predict_message("Congratulations! You've won a FREE iPhone. Click here now!")
# → 🚨 SPAM

predict_message("Hey, are we still meeting tomorrow at 5pm?")
# → ✅ HAM (Legitimate)
```

---

## 📉 Visualizations

- **Spam vs Ham Distribution** — Bar chart showing class balance
- **Message Length Distribution** — Histogram comparing lengths
- **Confusion Matrix** — Heatmap for both models

---

