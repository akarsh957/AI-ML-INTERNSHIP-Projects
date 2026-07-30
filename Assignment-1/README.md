# 🏥 Medical Insurance Cost Prediction using Multiple Linear Regression

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Objective

The goal of this project is to build a **Multiple Linear Regression** model that predicts individual **medical insurance charges** based on personal and demographic attributes such as age, BMI, smoking habits, number of dependents, and geographic region. The project demonstrates a complete supervised machine learning workflow — from data exploration to model evaluation.

---

## 📊 Dataset

- **Source:** [Medical Cost Personal Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Records:** 1,338
- **Features:** 7 (6 independent + 1 target)

| Feature    | Type        | Description                              |
|------------|-------------|------------------------------------------|
| `age`      | Numerical   | Age of primary beneficiary               |
| `sex`      | Categorical | Gender (male / female)                   |
| `bmi`      | Numerical   | Body Mass Index                          |
| `children` | Numerical   | Number of dependents                     |
| `smoker`   | Categorical | Smoking status (yes / no)                |
| `region`   | Categorical | US residential region (4 regions)        |
| `charges`  | Numerical   | Medical costs billed by insurance **(Target)** |

> **Note:** The dataset is **not included** in this repository. Download it from the Kaggle link above and place `insurance.csv` in the project root directory.

---

## 🛠️ Libraries Used

| Library         | Purpose                                     |
|-----------------|---------------------------------------------|
| `pandas`        | Data loading, manipulation, and analysis    |
| `numpy`         | Numerical computations                      |
| `matplotlib`    | Static data visualizations                  |
| `seaborn`       | Statistical data visualizations             |
| `scikit-learn`  | Model building, preprocessing, evaluation   |

---

## 🔄 Project Workflow

```
1. Data Loading & Exploration
        ↓
2. Data Cleaning (Missing values, Duplicates)
        ↓
3. Categorical Encoding (LabelEncoder)
        ↓
4. Exploratory Data Analysis (Histograms, Heatmap)
        ↓
5. Train-Test Split (80/20)
        ↓
6. Model Training (Multiple Linear Regression)
        ↓
7. Prediction on Test Data
        ↓
8. Model Evaluation (MAE, MSE, RMSE, R²)
        ↓
9. Visualization (Actual vs Predicted, Residuals)
        ↓
10. Conclusion & Observations
```

---

## 🤖 Model Used

**Multiple Linear Regression** from `sklearn.linear_model.LinearRegression`

The model learns the linear relationship:

```
charges = β₀ + β₁(age) + β₂(sex) + β₃(bmi) + β₄(children) + β₅(smoker) + β₆(region)
```

Where:
- `β₀` is the intercept
- `β₁...β₆` are learned coefficients for each feature

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual values |
| **MSE** (Mean Squared Error) | Average of squared errors — penalizes large errors |
| **RMSE** (Root Mean Squared Error) | Square root of MSE — in original dollar units |
| **R² Score** | Proportion of variance explained by the model (0 to 1) |

---

## 📈 Results

> **Note:** The exact metric values are generated dynamically when the notebook is executed. Run `Assignment-1.ipynb` to obtain the results.

### Key Findings:

- **Smoking status** is the strongest predictor of insurance charges, with the highest positive coefficient.
- **Age** and **BMI** are the next most influential factors.
- The model explains a significant portion of the variance in insurance costs.
- The model tends to underpredict for very high-cost individuals due to non-linear interactions.

### Visualizations Generated:

| Plot | Description |
|------|-------------|
| `images/histogram.png` | Distribution of all numerical features |
| `images/heatmap.png` | Correlation heatmap of all features |
| `images/actual_vs_predicted.png` | Scatter plot comparing actual vs predicted charges |
| `images/target_distribution.png` | Distribution of the target variable (`charges`) |
| `images/residuals.png` | Distribution of prediction residuals |

---

## 📝 Conclusion

This project demonstrates a complete machine learning pipeline for predicting medical insurance costs using Multiple Linear Regression. The analysis identified **smoking status** as the dominant predictor, followed by **age** and **BMI**. The model achieves a reasonable R² score, capturing the general pricing trend well. However, linear regression's inherent limitation — its inability to model non-linear relationships and feature interactions without explicit engineering — causes it to struggle with high-cost outliers. Future work could incorporate polynomial features, regularization (Ridge/Lasso), or ensemble methods like Random Forest to improve accuracy.

---

## 📁 Repository Structure

```
Assignment-1/
│
├── Assignment-1.ipynb      # Complete Jupyter Notebook with code & analysis
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies
│
└── images/                 # Generated visualizations (created by notebook)
    ├── actual_vs_predicted.png
    ├── heatmap.png
    ├── histogram.png
    ├── target_distribution.png
    └── residuals.png
```

---

## 🔗 Dataset Link

📎 **[Medical Cost Personal Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)**

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Assignment-1.git
   cd Assignment-1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) and place `insurance.csv` in the project root.

4. **Run the notebook:**
   ```bash
   jupyter notebook Assignment-1.ipynb
   ```

---

## 📄 License

This project is for educational purposes as part of an AI/ML internship assignment.

---

*Built with ❤️ using Python and scikit-learn*
