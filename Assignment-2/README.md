# 📉 Customer Churn Prediction using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Objective

The goal of this project is to build a **Logistic Regression** classifier that predicts whether a telecom customer will **churn** (cancel their subscription) based on demographic, account, and service usage attributes. The project demonstrates a complete supervised machine learning classification workflow — from data exploration and preprocessing to model evaluation and business-driven interpretation of results.

---

## 📊 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043
- **Features:** 21 (19 independent + 1 identifier + 1 target)

| Feature | Type | Description |
|---------|------|-------------|
| `customerID` | ID | Unique customer identifier (dropped before training) |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Numerical | Whether the customer is a senior citizen (0/1) |
| `Partner` | Categorical | Whether the customer has a partner |
| `Dependents` | Categorical | Whether the customer has dependents |
| `tenure` | Numerical | Number of months with the company |
| `PhoneService` | Categorical | Whether the customer has phone service |
| `MultipleLines` | Categorical | Whether the customer has multiple phone lines |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Whether the customer has online security |
| `OnlineBackup` | Categorical | Whether the customer has online backup |
| `DeviceProtection` | Categorical | Whether the customer has device protection |
| `TechSupport` | Categorical | Whether the customer has tech support |
| `StreamingTV` | Categorical | Whether the customer has streaming TV |
| `StreamingMovies` | Categorical | Whether the customer has streaming movies |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Whether the customer uses paperless billing |
| `PaymentMethod` | Categorical | Payment method used |
| `MonthlyCharges` | Numerical | Monthly charges billed ($) |
| `TotalCharges` | Numerical | Total charges billed ($) — contains whitespace quirk |
| `Churn` | Binary | Whether the customer churned (Yes / No) — **Target** |

> **Note:** The dataset is **not included** in this repository. Download it from the Kaggle link above and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root directory.

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, and analysis |
| `numpy` | Numerical computations |
| `matplotlib` | Static data visualizations |
| `seaborn` | Statistical data visualizations |
| `scikit-learn` | Model building, preprocessing, evaluation |

---

## 🔄 Methodology

```
1. Data Loading & Exploration
        ↓
2. Feature Type Identification (Numerical / Categorical / Target)
        ↓
3. Data Cleaning (TotalCharges whitespace → numeric coercion, NaN imputation)
        ↓
4. Drop Identifier Column (customerID)
        ↓
5. Encode Target Variable (Churn: Yes → 1, No → 0)
        ↓
6. One-Hot Encode Categorical Features (drop_first=True)
        ↓
7. Exploratory Visualizations (Correlation analysis, Distributions)
        ↓
8. Train-Test Split (80/20, stratified, random_state=42)
        ↓
9. Feature Scaling (StandardScaler on tenure, MonthlyCharges, TotalCharges)
        ↓
10. Model Training (Logistic Regression)
        ↓
11. Prediction on Test Data
        ↓
12. Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
        ↓
13. Feature Importance Analysis (Logistic Regression Coefficients)
        ↓
14. Observations & Conclusion
```

### Key Preprocessing Steps

- **TotalCharges Whitespace Fix:** The `TotalCharges` column contains whitespace strings (`' '`) for 11 newly onboarded customers with 0 tenure. These are coerced to numeric using `pd.to_numeric(errors='coerce')` and the resulting NaN values are imputed with 0.
- **One-Hot Encoding:** All categorical variables are encoded using `pd.get_dummies(drop_first=True)` to avoid the dummy variable trap.
- **Feature Scaling:** `StandardScaler` is applied to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to standardize their range, ensuring model convergence and fair coefficient comparison.

---

## 🤖 Model Used

**Logistic Regression** from `sklearn.linear_model.LogisticRegression`

The model predicts the probability of churn using the logistic function:

```
P(Churn = 1 | X) = 1 / (1 + e^(-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)))
```

Where:
- `β₀` is the intercept
- `β₁...βₙ` are learned coefficients for each feature
- Positive coefficients increase churn probability; negative coefficients decrease it

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correctly classified instances |
| **Precision** | Of all predicted churners, how many actually churned |
| **Recall** | Of all actual churners, how many were correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |

---

## 📈 Results

> **Note:** The exact metric values are generated dynamically when the notebook is executed. Run `Assignment-2.ipynb` to obtain the results.

| Metric | Score |
|--------|-------|
| **Accuracy** | ~0.80 |
| **Precision** | ~0.65 |
| **Recall** | ~0.54 |
| **F1-Score** | ~0.59 |

### Key Findings

- **Month-to-month contracts** are the strongest driver of churn — customers without long-term commitments are most likely to leave.
- **Fiber optic internet** users churn at higher rates, potentially due to higher costs or unmet service expectations.
- **Absence of tech support and online security** add-ons correlates strongly with churn, suggesting value-added services act as retention anchors.
- **Longer tenure** and **two-year contracts** are the strongest churn preventers.

### Visualizations Generated

| Plot | Description |
|------|-------------|
| `images/churn_distribution.png` | Distribution of the target variable (Churn) |
| `images/correlation_with_churn.png` | Top 15 features correlated with churn |
| `images/numerical_distributions.png` | Distribution of numerical features by churn status |
| `images/confusion_matrix.png` | Confusion matrix heatmap |
| `images/feature_coefficients.png` | Top positive and negative logistic regression coefficients |

---

## 📝 Conclusion

The Logistic Regression model achieves ~80% accuracy but reveals a precision-recall trade-off critical for customer retention: while precision (~65%) ensures most flagged churners are genuine, the recall (~54%) means nearly half of actual churners go undetected. The coefficient analysis identifies **month-to-month contracts**, **fiber optic internet**, and **lack of tech support/online security** as the top churn drivers — actionable insights for targeted retention campaigns. A key limitation of Logistic Regression is its assumption of **linearity between features and log-odds**, which prevents it from capturing complex interaction effects without manual feature engineering. Future improvements could include threshold tuning, class-weight balancing, or ensemble methods (Random Forest, XGBoost) to improve recall.

---

## 📁 Repository Structure

```
Assignment-2/
│
├── Assignment-2.ipynb      # Complete Jupyter Notebook with code & analysis
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies
│
└── images/                 # Generated visualizations (created by notebook)
    ├── churn_distribution.png
    ├── correlation_with_churn.png
    ├── numerical_distributions.png
    ├── confusion_matrix.png
    └── feature_coefficients.png
```

---

## 🔗 Dataset Link

📎 **[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Assignment-2.git
   cd Assignment-2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.

4. **Run the notebook:**
   ```bash
   jupyter notebook Assignment-2.ipynb
   ```

---

## 📄 License

This project is for educational purposes as part of an AI/ML internship assignment.

---

*Built with ❤️ using Python and scikit-learn*
