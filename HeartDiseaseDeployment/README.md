# Heart Disease Risk Assessment System

This repository contains the complete implementation for **AI-ML Assignment 10: End-to-End Machine Learning Model Deployment using GitHub and Render**. The system trains a Random Forest Classifier on patient clinical data, exposes the model via a Flask REST API, and provides a polished web interface for running real-time predictions.

---

## 🚀 Live Render Deployment URL

You can access the live deployed web service here:
👉 **[INSERT YOUR RENDER DEPLOYMENT URL HERE]** *(e.g., `https://heart-disease-deployment.onrender.com`)*

---

## 📁 Repository Structure

```text
HeartDiseaseDeployment/
├── heart.csv               # Dataset loaded from Kaggle / public source
├── train_model.py          # Preprocesses data, trains & evaluates the model
├── app.py                  # Flask API & web service dashboard
├── requirements.txt        # Production dependencies for deployment
├── README.md               # Project documentation & assignment conclusion
├── model.pkl               # Trained & serialized classifier (generated)
└── templates/              # Directory containing HTML layout templates
    └── index.html          # Dynamic web page dashboard
```

---

## 🛠️ Local Development Guide (Windows CMD)

Follow these instructions to run the project locally on your machine using Windows Command Prompt (CMD).

### 1. Project Setup
Open CMD, navigate to your workspace folder, and set up a virtual environment:
```cmd
:: Navigate to the project directory
cd /d "d:\AI ML INTERNSHIP Projects\HeartDiseaseDeployment"

:: Create a Python virtual environment
python -m venv venv

:: Activate the virtual environment
venv\Scripts\activate

:: Install required dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data and Train the Model
Run the training script to load the dataset, verify features, display sample rows, check for missing values, evaluate test accuracy, and save the serialized model:
```cmd
python train_model.py
```
*Expected Output:*
- Prints Task 1 preprocessing steps and shape.
- Trains Random Forest Classifier.
- Displays Test Accuracy Score (approx. `100.00%`).
- Creates `model.pkl` in the root folder.

### 3. Run the Flask Web API Locally
Start the local server using:
```cmd
python app.py
```
*Expected Output:*
- The server starts on `http://127.0.0.1:5000/`.
- Open your browser and navigate to `http://127.0.0.1:5000/` to use the interactive Web UI.

### 4. Test the API Endpoint (Command Line)
To send a POST request with sample patient details to the `/predict` API, open another Command Prompt window or a PowerShell terminal and run:

**Using PowerShell (CMD-accessible):**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"age":58,"sex":1,"cp":0,"trestbps":150,"chol":270,"fbs":1,"restecg":0,"thalach":111,"exang":1,"oldpeak":3.0,"slope":1,"ca":0,"thal":3}'
```

**Using curl (Windows CMD):**
```cmd
curl -X POST -H "Content-Type: application/json" -d "{\"age\":58,\"sex\":1,\"cp\":0,\"trestbps\":150,\"chol\":270,\"fbs\":1,\"restecg\":0,\"thalach\":111,\"exang\":1,\"oldpeak\":3.0,\"slope\":1,\"ca\":0,\"thal\":3}" http://127.0.0.1:5000/predict
```

*Expected JSON Response:*
```json
{
  "prediction": "Heart Disease Detected"
}
```

---

## 🐙 Git and GitHub Pushing Instructions

Use the following commands in Windows CMD to initialize a local Git repository and push all source files, the trained model, and the dataset to your public GitHub repository:

```cmd
:: Initialize a local Git repository
git init

:: Create a .gitignore file to exclude virtual environments and cache
echo venv/ > .gitignore
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore

:: Stage all files in the project directory
git add .

:: Commit files locally
git commit -m "Initial commit - Heart Disease ML Deployment files"

:: Set branch to main
git branch -M main

:: Link local repository to your public GitHub repo
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/HeartDiseaseDeployment.git

:: Push repository code to GitHub (requires GitHub login on first run)
git push -u origin main
```

---

## ☁️ Render Deployment Steps

1. Create a free account at [Render](https://render.com/).
2. Click **New +** and choose **Web Service**.
3. Link your GitHub account and select your public `HeartDiseaseDeployment` repository.
4. Configure the service settings:
   - **Name:** `heart-disease-risk-prediction`
   - **Language/Environment:** `Python 3`
   - **Region:** Choose the closest region (e.g., Singapore, Oregon).
   - **Branch:** `main`
   - **Root Directory:** Keep blank (or specify `HeartDiseaseDeployment` if nested).
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Select the **Free instance type** and click **Deploy Web Service**.
6. Copy the provided live service URL from Render and replace the placeholder at the top of this README.

---

## 📝 Conclusion

This project successfully developed and deployed a Random Forest Classifier for heart disease risk prediction, achieving a 100.00% accuracy score on the test set. While the model showed perfect split performance, real-world deployment on cloud platforms like Render introduces key challenges, including managing OS-level library dependencies, handling container spin-up delays for free tier services, and ensuring API input validation. Resolving these challenges highlights the critical importance of MLOps. MLOps goes beyond building models; it establishes robust version control, automated testing, continuous integration, and seamless API serving. Implementing MLOps practices ensures that machine learning models do not remain isolated scripts but function as reliable, scalable, and observable software services in production environments, bridging the gap between data science and software engineering.
