from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd

app = FastAPI(title="Appliances Energy Prediction API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scaler on startup
models = {}
scaler = None
feature_names = []

@app.on_event("startup")
def load_artifacts():
    global models, scaler, feature_names
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    try:
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
        
        # Load the 4 models
        for model_name in ["KNN", "Decision_Tree", "Linear_Regression", "Random_Forest"]:
            models[model_name] = joblib.load(os.path.join(models_dir, f"{model_name}.joblib"))
            
        print("Models and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

class PredictionRequest(BaseModel):
    model: str # "KNN", "Decision_Tree", "Linear_Regression"
    features: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to the Appliances Energy Prediction API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found. Available models: {list(models.keys())}")
    
    model = models[request.model]
    
    # Extract features in the correct order
    try:
        input_data = []
        for feature in feature_names:
            if feature not in request.features:
                raise HTTPException(status_code=400, detail=f"Missing feature: {feature}")
            input_data.append(float(request.features[feature]))
            
        # Convert to 2D numpy array
        input_array = np.array([input_data])
        
        # Scale
        input_scaled = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(input_scaled)
        
        return {"prediction": float(prediction[0])}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

