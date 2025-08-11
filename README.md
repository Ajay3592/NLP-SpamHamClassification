# 📧 Spam vs Ham Classification

This project focuses on building a robust machine learning pipeline to classify messages as **spam** or **ham** (non-spam). The dataset used was **imbalanced**, and various techniques were applied to improve model performance and fairness across classes.

---

## 🚀 Project Overview

- **Goal:** Classify text messages as spam or ham
- **Challenge:** Imbalanced dataset with more ham than spam
- **Solution:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the data
- **Models Used:**  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
  - XGBoost  
  - Logistic Regression

---

## 🧠 Preprocessing Steps

- Text cleaning (lowercasing, punctuation removal, etc.)
- Tokenization
- Stopword removal
- Feature extraction using TF-IDF
- **Stemming** and/or **Lemmatization** (based on experimentation)
- Handling class imbalance using **SMOTE**

---

## 📊 Evaluation Metrics

While **accuracy** was consistently above **95%** for most models, further evaluation was done using:

- **Precision**
- **Recall**
- **F1 Score** (to account for class imbalance)

---

## 📈 Results Summary

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| SVM                 | 95%+     | High     |
| Random Forest       | 95%+     | High     |
| XGBoost             | 95%+     | High     |
| Logistic Regression | 95%+     | High     |
| K-Nearest Neighbors | **51%**  | Low      |

> ⚠️ Note: KNN underperformed significantly, likely due to its sensitivity to feature scaling and high-dimensional sparse data from TF-IDF. It was retained for comparison purposes.

---

## 🧪 Imbalance Handling

- **SMOTE** was used to synthetically generate samples for the minority class (spam).
- This helped improve recall and F1 score without sacrificing accuracy in most models.

---

## 🛠️ Technologies Used

- Python
- scikit-learn
- imbalanced-learn
- XGBoost
- NLTK / spaCy
- Pandas, NumPy, Matplotlib, Seaborn

---

## 📌 Future Improvements

- Experiment with deep learning models (e.g., LSTM, BERT)
- Integrate real-time spam detection API
- Explore ensemble methods for further boosting performance
- Investigate dimensionality reduction or feature selection to improve KNN
