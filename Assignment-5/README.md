# Employee Attrition Prediction using Decision Tree and Random Forest Classification

## Objective
The objective of this project is to build and compare predictive models to identify employees who are likely to leave an organization (attrition). By leveraging machine learning classification techniques (specifically Decision Trees and Random Forests), we aim to provide HR departments with data-driven insights to proactively address retention challenges and optimize employee engagement.

## Dataset Link
The dataset used in this project is the **IBM HR Analytics Employee Attrition & Performance Dataset**, which is publicly available on Kaggle:
* **Kaggle Link:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
* *Note:* In compliance with licensing and data usage rules, the dataset file is not hosted directly in this repository.

## Libraries Used
The project is implemented in Python and utilizes the following data science and machine learning libraries:
* **Pandas:** For data loading, manipulation, and exploratory data analysis.
* **NumPy:** For numerical operations.
* **Scikit-Learn:** For data preprocessing, model building (Decision Tree and Random Forest), and performance evaluation.
* **Matplotlib:** For static visualizations.
* **Seaborn:** For advanced statistical graphics and heatmaps.

## Methodology
The classification task is executed through a structured pipeline:
1. **Data Understanding:** Loaded the dataset of 1,470 records and 35 columns, programmatically separating features by data type (numerical vs. categorical) and displaying baseline summary statistics.
2. **Data Preprocessing:**
   * Checked for missing values (none were present).
   * Dropped zero-variance and administrative identifier columns: `EmployeeCount`, `EmployeeNumber`, `Over18`, and `StandardHours`.
   * Encoded the binary target variable `Attrition` ('Yes' $\rightarrow$ 1, 'No' $\rightarrow$ 0).
   * One-hot encoded the remaining categorical columns using `pd.get_dummies(drop_first=True)` to prevent the dummy variable trap.
   * Split the dataset into 80% training and 20% testing sets using stratified splitting (`stratify=y`) to maintain the target class proportion.
3. **Model Development:**
   * Trained a single Decision Tree Classifier (`random_state=42`).
   * Trained an ensemble Random Forest Classifier (`n_estimators=100`, `random_state=42`).
4. **Evaluation:** Evaluated both models using test set classification metrics (Accuracy, Precision, Recall, and F1-Score), plotted side-by-side Confusion Matrices, and analyzed feature importances.
5. **Bonus Hyperparameter Tuning:** Tested `max_depth` hyperparameter values (`None`, `10`, `5`) on Random Forest to analyze regularization effects.

## Results & Model Comparison

The following table summarizes the evaluation metrics obtained on the 20% test set ($N=294$):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | 76.53% | 31.03% | 38.30% | 34.29% |
| **Random Forest Classifier** (Default, Depth=None) | 83.33% | 41.67% | 10.64% | 16.95% |
| **Random Forest Classifier** (Tuned, Depth=10) | **83.67%** | **45.45%** | 10.64% | **17.24%** |
| **Random Forest Classifier** (Tuned, Depth=5) | 82.99% | 33.33% | 6.38% | 10.71% |

## Key Observations
1. **Ensemble Generalization:** The Random Forest Classifier achieved significantly higher overall accuracy (~83.3%) compared to the Decision Tree (~76.5%). This demonstrates the strength of ensemble methods in reducing overfitting on unseen test data.
2. **Precision vs. Recall Tradeoff:** The Random Forest model achieved higher precision (41.67% vs. 31.03% for Decision Tree), meaning its positive predictions are more reliable. However, the Decision Tree achieved a higher recall (38.30% vs. 10.64%), capturing a larger portion of employees who actually left, albeit at the cost of a higher false-positive rate.
3. **Class Imbalance Effect:** The extreme recall drop in the Random Forest is a consequence of the severe class imbalance in the dataset (only ~16% positive instances). Because the default Random Forest model optimizes for overall accuracy, it heavily biases its predictions toward the majority class (No Attrition).
4. **Regularization through Depth Tuning:** Tuning the `max_depth` of Random Forest to `10` yielded the best results, improving Accuracy to 83.67% and Precision to 45.45%. Restricting the depth prevents the individual trees from growing too deep and memorizing noise, thereby improving generalizability. Pruning too much (`max_depth=5`), however, leads to underfitting.

## Conclusion
The **Random Forest Classifier** performed better overall due to its ensemble nature, which aggregates predictions from multiple decision trees. By using bootstrap bagging and feature subspace sampling, Random Forests reduce variance and prevent the overfitting that commonly plagues individual Decision Trees, resulting in superior generalization and higher precision.

* **Key Limitation of Decision Trees:** High variance. Single decision trees are highly sensitive to training data fluctuations; a small change in the training dataset can yield a completely different tree structure, making them prone to overfitting.
* **Key Limitation of Random Forests:** Computational complexity and lack of interpretability. Building and scoring an ensemble of hundreds of trees requires significantly more memory and CPU cycles than a single tree, and the aggregated decision path behaves like a "black box" that is difficult to visually explain to business stakeholders.
