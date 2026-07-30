# Assignment 4: Breast Cancer Classification using K-Nearest Neighbors (KNN)

This repository contains a complete, production-grade implementation of a machine learning workflow to classify breast tumors as **Malignant (M)** or **Benign (B)** using the **K-Nearest Neighbors (KNN)** algorithm.

---

## 🎯 Objective
Develop a K-Nearest Neighbors classification model using diagnostic features extracted from cell nuclei to predict tumor malignancy. The clinical focus is on maximizing model reliability while highlighting the significance of performance metrics like **Recall** in medical diagnosis to minimize critical false negatives.

---

## 📊 Dataset
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* **Target Variable:** `diagnosis` (`M` = Malignant, `B` = Benign)
* **Instance Count:** 569 instances
* **Feature Count:** 30 real-valued numerical features (e.g., radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry)

---

## 🛠️ Libraries Used
The project relies on standard scientific Python libraries:
* **Pandas:** For loading, cleaning, and manipulating structured tabular data.
* **NumPy:** For vector operations and numerical computations.
* **Scikit-learn:** For data preprocessing (`StandardScaler`, `train_test_split`), model building (`KNeighborsClassifier`), and performance evaluation (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`).
* **Matplotlib & Seaborn:** For visual analysis and plotting the confusion matrix heatmap.

---

## ⚙️ Methodology

1. **Data Understanding:**
   * Load raw `data.csv`.
   * Inspect structural features using `.head()`, `.info()`, and `.describe()`.
   * Programmatically separate and identify the 30 numerical feature columns.
   * Analyze target class distributions (357 Benign vs. 212 Malignant).

2. **Data Preprocessing:**
   * **Data Cleaning:** Inspect and handle missing values, and drop non-informative metadata columns (`id` and the trailing empty `Unnamed: 32` column).
   * **Target Encoding:** Map the classification labels (`M` $\rightarrow$ 1, `B` $\rightarrow$ 0) to convert them to binary numeric outputs.
   * **Feature Scaling:** Apply `StandardScaler` to standardize all feature distributions to have a mean of 0 and standard deviation of 1.
     > *Note:* Feature scaling is **mandatory** for distance-based algorithms like KNN. Because KNN relies on calculating distance metrics (e.g., Euclidean distance) between data points, unscaled features with large ranges (like `area_mean` which can exceed 2000) would dominate the distance calculation over features with small scales (like `smoothness_mean` which is $<0.2$), introducing significant model bias.
   * **Data Splitting:** Partition the dataset into an 80% training set (455 samples) and a 20% testing set (114 samples). The split is **stratified** based on the target variable to preserve original class distributions.

3. **Model Development:**
   * Instantiate a `KNeighborsClassifier` with $K=5$ neighbors.
   * Fit/train the model on the scaled training features and target labels.
   * Query the trained model to predict class labels for the test dataset.

4. **Model Evaluation:**
   * Calculate accuracy, precision, recall, and F1-score on the test dataset.
   * Generate a confusion matrix and plot a Seaborn heatmap representing predictions.

---

## 🏆 Results & Metrics

Below are the performance metrics achieved by the K-Nearest Neighbors ($K=5$) model on the test dataset (20% of samples):

| Metric | Score | Percentage | Clinical Significance / Definition |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 0.9561 | 95.61% | Overall fraction of correct predictions across all test cases. |
| **Precision** | 0.9744 | 97.44% | Fraction of predicted malignant tumors that are actually malignant. Minimizes false alarms/unnecessary biopsies. |
| **Recall (Sensitivity)** | 0.9048 | 90.48% | **Critical for Healthcare:** The percentage of actual malignant cases correctly detected. Higher recall minimizes **False Negatives** (missed cancers). |
| **F1-Score** | 0.9383 | 93.83% | Harmonic mean of Precision and Recall, proving a balanced classifier performance. |

### Confusion Matrix Breakdown
* **True Negatives (TN):** 71 (Benign cases correctly predicted as Benign)
* **False Positives (FP):** 1 (Benign case incorrectly predicted as Malignant)
* **False Negatives (FN):** 4 (Malignant cases incorrectly predicted as Benign)
* **True Positives (TP):** 38 (Malignant cases correctly predicted as Malignant)

---

## 📌 Conclusion

### 1. Key Findings
The model successfully classifies tumor samples with high precision (**97.44%**) and strong accuracy (**95.61%**). However, the recall of **90.48%** indicates that 4 out of 42 malignant tumors in the test set were missed (classified as benign). In clinical workflows, even a small number of False Negatives can have life-threatening implications due to delayed intervention. Therefore, this model is highly suited as a preliminary screening aid (triage) but requires threshold tuning or diagnostic validation before serving as a standalone diagnostic decision system.

### 2. Importance of Feature Scaling in KNN
KNN determines proximity based on distance calculations in a multi-dimensional space. Features on large ranges (like tumor area) skew calculations, rendering smaller range features (like symmetry or smoothness) negligible. Standardization via `StandardScaler` places all 30 features on an equivalent scale (mean=0, variance=1), allowing each medical feature to contribute equally to the neighbor lookup.

### 3. One Limitation of KNN
A primary limitation of KNN is its **high computational latency during inference/prediction (lazy learning)**. Unlike eager algorithms that build a generalized decision boundary during training (such as Logistic Regression or Support Vector Machines), KNN does not construct a model. It simply stores the training set. For every test observation, KNN must calculate distances to all $N$ training examples and sort them to locate the nearest neighbors, resulting in a time complexity of $O(N \times D)$ per prediction. This makes the algorithm highly inefficient as the database size grows.
