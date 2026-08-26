import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# This is for Vercel serverless function
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(str("Breast Cancer Prediction API is running!").encode())
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse JSON data
            data = json.loads(post_data)
            
            # Get features from request
            features = data.get('features', [])
            
            if not features:
                response = {"error": "No features provided"}
                self._send_response(400, response)
                return
            
            # Train model if not exists
            model, scaler, feature_names = train_model()
            
            # Convert to numpy array
            input_data = np.array(features).reshape(1, -1)
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            # Prepare response
            result = {
                "prediction": int(prediction),
                "diagnosis": "Malignant" if prediction == 1 else "Benign",
                "confidence": float(max(probabilities)),
                "probability_benign": float(probabilities[0]),
                "probability_malignant": float(probabilities[1])
            }
            
            self._send_response(200, result)
            
        except Exception as e:
            response = {"error": str(e)}
            self._send_response(500, response)
    
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def train_model():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, data.feature_names