# Weather Condition Classification using Support Vector Machine (SVM)

This repository contains a complete, executable Python and Jupyter Notebook implementation for an academic machine learning assignment. The goal is to build a binary classification model that categorizes weather conditions into **Warm** ($\ge 25^{\circ}\text{C}$) and **Cool** ($< 25^{\circ}\text{C}$) based on meteorological features retrieved from the Open-Meteo Weather API.

---

## Objective
The objective of this project is to collect hourly weather data from the public Open-Meteo API, preprocess and clean the dataset, standardise features, train a Support Vector Machine (SVM) Classifier using the Radial Basis Function (RBF) kernel, and evaluate its classification performance. 

---

## API Documentation Link
This project retrieves real-time weather forecasting data from the:
* **Open-Meteo Weather API:** [https://open-meteo.com/](https://open-meteo.com/)

---

## Libraries Used
* **Data Retrieval:** `requests`
* **Data Processing & Analysis:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`

---

## Methodology

The project is executed in a sequential machine learning workflow as detailed below:
1. **Data Collection & Understanding:**
   * Hourly weather data for Bangalore, India (`latitude=12.9716`, `longitude=77.5946`) was fetched using the `requests` library. Bangalore was chosen because its late July temperate weather provides a balanced distribution of hours above and below the $25^{\circ}\text{C}$ threshold.
   * The JSON payload containing temperature, relative humidity, surface pressure, and wind speed was extracted and converted into a Pandas DataFrame.
   * The binary target column `Weather_Class` was programmatically generated based on the threshold: **Warm** ($\ge 25^{\circ}\text{C}$) and **Cool** ($< 25^{\circ}\text{C}$).
2. **Data Preprocessing:**
   * Handled missing values (verified 0 null values in the API payload).
   * Removed non-predictive time columns to focus exclusively on meteorological features.
   * Encoded the binary target variable numerically (`Warm` -> 1, `Cool` -> 0).
   * Split the dataset into 80% training (134 records) and 20% testing (34 records) datasets using a stratified split (`random_state=42`, `stratify=y`) to maintain class ratios.
   * Standardised all input features (`temperature_2m`, `relative_humidity_2m`, `surface_pressure`, `wind_speed_10m`) using `StandardScaler` to have a mean of 0 and variance of 1, fitting the scaler **only** on the training set.
3. **Model Development:**
   * Initialised the Support Vector Classifier (`SVC`) with a Radial Basis Function (`rbf`) kernel and `random_state=42`.
   * Trained the model on the scaled training features and predicted classes on the scaled test set.
4. **Model Evaluation:**
   * Computed Accuracy, Precision, Recall, and F1-Score.
   * Visualised class separation performance using a Seaborn-based confusion matrix heatmap.

---

## Results

### Model Performance Metrics on Test Set
The model's classification performance on the test set is summarized in the table below:

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | 0.9706 | 97.06% |
| **Precision** | 1.0000 | 100.00% |
| **Recall** | 0.9167 | 91.67% |
| **F1-Score** | 0.9565 | 95.65% |

### Confusion Matrix Summary
The evaluation test set contains 34 total records (22 Cool, 12 Warm). The confusion matrix details are:
* **True Negatives (TN):** 22 (Cool classified as Cool)
* **False Positives (FP):** 0 (Cool classified as Warm)
* **False Negatives (FN):** 1 (Warm classified as Cool)
* **True Positives (TP):** 11 (Warm classified as Warm)

The model achieved a precision of **100%**, which indicates that there are no false positives; every prediction of a warm condition was correct. The recall of **91.67%** indicates that only 1 warm hour was misclassified as cool (false negative).

---

## Key Observations

1. **High Overall Classification Accuracy:** The model achieved an outstanding classification accuracy of **97.06%** on the test set, demonstrating that the combined meteorological features (`temperature_2m`, `relative_humidity_2m`, `surface_pressure`, and `wind_speed_10m`) contain a very strong predictive signal for classifying weather conditions.
2. **Perfect Precision Score (100%):** The model obtained a precision of **100.00%** with zero false positives ($FP=0$), which means every single instance predicted as 'Warm' was indeed a 'Warm' hour. The decision boundary successfully isolates the 'Warm' class without misclassifying any 'Cool' hours as 'Warm'.
3. **High Recall with Minor Boundary Misclassification:** The model achieved a recall of **91.67%** due to a single false negative ($FN=1$), where one 'Warm' hour was misclassified as 'Cool'. This indicates the model is highly sensitive in capturing warm conditions, with minor boundary ambiguity that slightly favors 'Cool' classifications near the $25^{\circ}\text{C}$ threshold.

---

## Conclusion
* **Key Findings:** This experiment demonstrates that an SVM classifier with an RBF kernel can highly accurately predict weather condition classes based on real-time features retrieved from the Open-Meteo API. The pipeline is robust, yielding an overall accuracy of 97.06% and perfect precision.
* **Critical Importance of Feature Scaling:** Feature scaling using `StandardScaler` is absolutely mandatory for SVM. Because SVM works by maximizing the geometric margin between support vectors using Euclidean distance, features with larger scales (like `surface_pressure` $\approx 912\text{ hPa}$) would dominate features with smaller scales (like `wind_speed_10m` $\approx 10\text{ km/h}$), biasing the margin boundary. Scaling normalizes the contribution of all features.
* **Advantages and Limitations of SVM:**
  * *Advantage:* Capable of modeling complex, non-linear decision boundaries through the kernel trick (e.g., using the RBF kernel) without explicitly projecting data into high-dimensional space.
  * *Limitation:* The algorithm does not scale well to large datasets as its training complexity is between $O(N^2)$ and $O(N^3)$, and it is highly sensitive to the choice of hyperparameters ($C$ and $\gamma$).