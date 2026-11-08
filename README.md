# Machine Learning — Assignment 2

* **Name:** Nivedita Shinde

---

## a. Problem statement
The objective of this assignment is to develop, evaluate, and benchmark five distinct machine learning classification algorithms on a standardized diagnostic dataset, and deploy an interactive web application using Streamlit Community Cloud. The application allows users to upload held-out test data, select any of the trained classification models, and view real-time evaluation metrics, confusion matrices, and classification reports without retraining the models.

---

## b. Dataset description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source Repository:** UCI Machine Learning Repository
* **UCI Repository URL:** [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic]
* **Number of Instances:** 569 instances (satisfies the minimum requirement of $\ge$ 500)
* **Number of Features:** 30 continuous real-valued features computed from digitized images of fine needle aspirate (FNA) of breast mass (satisfies the minimum requirement of $\ge$ 12)
* **Target Attribute:** `target` (Binary Classification: `0` = Malignant, `1` = Benign)
* **Data Preprocessing & Splitting:**
  * Stratified 80:20 Train-Test split performed to preserve target class proportions (Training set: 455 instances, Testing set: 114 instances).
  * `StandardScaler` fitted strictly on the training set features and saved as `model/scaler.pkl` to prevent data leakage.
  * `test_data.csv` exported containing all 30 input features alongside the ground-truth `target` column.

---

## c. Github Repository Link
* **GitHub Repository Link:** https://github.com/NiveditaMShinde/breast-cancer-diagnostic-classifier
* **Live Streamlit App Link:** https://breast-cancer-diagnostic-classifier-5mymsve5ftbivkwqqkbatr.streamlit.app/

---

## d. Models used:

### Model Evaluation Comparison Table
The evaluation metrics calculated on the held-out test dataset (`test_data.csv`) across all five implemented models are summarized below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9386 | 0.9342 | 0.9577 | 0.9444 | 0.9510 | 0.8689 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*(Metrics computed: Accuracy, ROC-AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient - MCC).*

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest Accuracy (98.25%) and top AUC (0.9954) alongside SVM. Feature standardization ensured stable, well-conditioned gradient updates across all 30 dimensions, yielding an outstanding MCC of 0.9623. |
| **Decision Tree** | Produced the lowest overall AUC (0.9342) and an Accuracy of 93.86%. Its orthogonal, axis-parallel splitting mechanism struggles to smoothly capture diagonal boundaries in continuous, high-dimensional biological feature spaces. |
| **KNN** | Delivered strong performance (Accuracy: 95.61%, AUC: 0.9788, MCC: 0.9054). Standard scaling was essential to prevent features with larger absolute magnitudes from dominating Euclidean distance calculations. |
| **Naive Bayes** | Maintained solid discrimination with an AUC of 0.9878 despite the naive class-conditional independence assumption, demonstrating that Gaussian approximations model the distribution of cell measurements reasonably well. |
| **Support Vector Machine** | Tied for the highest Accuracy (98.25%), F1 Score (0.9861), and MCC (0.9623). The soft-margin RBF kernel established an effective non-linear separation boundary between malignant and benign tumor profiles. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the overall winner for this dataset. In addition to tying for the best Accuracy (98.25%), F1 (0.9861), and MCC (0.9623), it achieved the highest AUC (0.9954) while providing superior computational efficiency and direct linear interpretability. |

---

## e. Project Repository Structure
```text
.
├── app.py                             # Interactive Streamlit frontend web application
├── requirements.txt                   # Deployment dependencies
├── README.md                          # Mandatory project documentation and metrics
├── test_data.csv                      # Held-out 20% test dataset with 'target' column
├── breast_cancer_classifier.ipynb     # Complete model training and evaluation notebook
└── model/                             # Serialized trained model and scaler artifacts
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── support_vector_machine.pkl