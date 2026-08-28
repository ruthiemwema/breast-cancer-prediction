import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib
import numpy as np


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the trained model and supporting files
MODEL_PATH = BASE_DIR / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)


class handler(BaseHTTPRequestHandler):

    def send_json(self, status_code, data):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_json(200, {"message": "OK"})

    def do_GET(self):
        self.send_json(
            200,
            {
                "message": "Breast Cancer Prediction API is running!",
                "features": len(feature_names),
                "endpoint": "/api/predict"
            }
        )

    def do_POST(self):

        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            # Convert JSON to Python dictionary
            data = json.loads(post_data.decode("utf-8"))

            # Get feature values
            features = data.get("features")

            if features is None:
                self.send_json(
                    400,
                    {"error": "No features were provided."}
                )
                return

            # Make sure exactly 30 measurements were provided
            if len(features) != len(feature_names):
                self.send_json(
                    400,
                    {
                        "error": f"Expected {len(feature_names)} features, "
                                 f"but received {len(features)}."
                    }
                )
                return

            # Convert input to NumPy array
            input_data = np.array(features, dtype=float).reshape(1, -1)

            # Scale using the SAME scaler used during training
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]

            # Get probabilities
            probabilities = model.predict_proba(input_scaled)[0]

            # Your project uses:
            # 0 = Benign
            # 1 = Malignant

            benign_probability = float(probabilities[0])
            malignant_probability = float(probabilities[1])

            diagnosis = (
                "Malignant"
                if prediction == 1
                else "Benign"
            )

            confidence = max(
                benign_probability,
                malignant_probability
            )

            result = {
                "prediction": int(prediction),
                "diagnosis": diagnosis,
                "confidence": float(confidence),
                "probability_benign": benign_probability,
                "probability_malignant": malignant_probability
            }

            self.send_json(200, result)

        except ValueError:
            self.send_json(
                400,
                {"error": "Feature values must be numeric."}
            )

        except Exception as e:
            self.send_json(
                500,
                {"error": str(e)}
            )