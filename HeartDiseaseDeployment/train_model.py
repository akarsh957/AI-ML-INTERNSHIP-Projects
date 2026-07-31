import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # -------------------------------------------------------------
    # Task 1: Data Understanding and Preprocessing
    # -------------------------------------------------------------
    print("=== TASK 1: Data Understanding and Preprocessing ===")
    
    # 1. Load the dataset using Pandas
    dataset_path = os.path.join(os.path.dirname(__file__), "heart.csv")
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 2. Display the first five records
    print("\n--- First Five Records ---")
    print(df.head())
    
    # 3. Identify Numerical features and the Target variable
    print("\n--- Identifying Features and Target ---")
    # Identify continuous numerical features vs categorical/binary features
    continuous_numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    target_variable = "target"
    
    print(f"Continuous Numerical Features: {continuous_numerical_features}")
    print(f"Categorical/Binary Features (represented numerically): {categorical_features}")
    print(f"Target Variable: '{target_variable}' (0 = No Heart Disease, 1 = Heart Disease Detected)")
    
    # 4. Check for missing values
    print("\n--- Checking for Missing Values ---")
    missing_values = df.isnull().sum()
    print(missing_values)
    total_missing = missing_values.sum()
    print(f"Total missing values in the dataset: {total_missing}")
    
    # 5. Split the dataset into 80% training and 20% testing
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    # Split into 80% train and 20% test with a random state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("\n--- Train-Test Split Details ---")
    print(f"Total dataset shape: {df.shape}")
    print(f"Training features shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing features shape: {X_test.shape}, Testing labels shape: {y_test.shape}")
    
    # -------------------------------------------------------------
    # Task 2: Model Development and Serialization
    # -------------------------------------------------------------
    print("\n=== TASK 2: Model Development ===")
    
    # Build a classification model (Random Forest Classifier is chosen for its robustness)
    print("Training Random Forest Classifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model using the Accuracy Score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Training Completed. Evaluation Metrics:")
    print(f"Test Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Save the trained model using joblib
    model_save_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    print(f"\nSaving the trained model to: {model_save_path}")
    joblib.dump(model, model_save_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
