import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    model = None
    print("WARNING: model.pkl not found. Please run train_model.py first.")

# List of features in the exact order the model was trained on
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@app.route("/", methods=["GET"])
def home():
    # Attempt to render interactive web interface if index.html exists, otherwise return HTML status
    try:
        return render_template("index.html")
    except Exception:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Heart Disease Risk Prediction API</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    color: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    background: rgba(255, 255, 255, 0.1);
                    padding: 40px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.18);
                    max-width: 500px;
                }
                h1 { margin-bottom: 10px; font-size: 2.2em; }
                p { color: #e0e0e0; font-size: 1.1em; line-height: 1.6; }
                .status {
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #4caf50;
                    border-radius: 30px;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Heart Disease Risk Prediction</h1>
                <p>Flask REST API is online and running successfully on Render!</p>
                <p>Send patient clinical parameters via a POST request to <code>/predict</code> to get results.</p>
                <div class="status">API STATUS: ONLINE</div>
            </div>
        </body>
        </html>
        """

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded on the server."}), 500
    
    # Parse JSON request data
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. Expected JSON body."}), 400
    
    # Validate and extract input features
    input_features = []
    missing_fields = []
    
    for feature in FEATURES:
        if feature not in data:
            missing_fields.append(feature)
        else:
            try:
                # Convert features to appropriate types (float for continuous, int for discrete if needed)
                # Using float handles both int and decimal inputs gracefully
                val = float(data[feature])
                input_features.append(val)
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid data type for field '{feature}'. Must be a number."}), 400
                
    if missing_fields:
        return jsonify({
            "error": "Missing required patient parameters.",
            "missing_fields": missing_fields
        }), 400
        
    try:
        # Prepare the input for model prediction as a DataFrame to keep feature names
        prediction_input = pd.DataFrame([input_features], columns=FEATURES)
        prediction_code = int(model.predict(prediction_input)[0])
        
        # Format the output prediction exactly as requested:
        # {"prediction": "Heart Disease Detected"}
        if prediction_code == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"
            
        return jsonify({"prediction": result})
        
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    # Host on 0.0.0.0 and dynamically bind port for Render deployment compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
