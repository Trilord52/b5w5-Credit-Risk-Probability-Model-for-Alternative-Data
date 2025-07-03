import mlflow.pyfunc
import logging
import numpy as np
import pandas as pd
from typing import List
from src.api.pydantic_models import CustomerFeatures

# Load the trained model and scaler from the models/ directory
try:
    model = mlflow.pyfunc.load_model(model_uri="models:/best_model/Production")
    logging.info("Loaded model from MLflow registry.")
except Exception as e:
    logging.warning(f"MLflow model load failed: {e}. Falling back to local file.")
    import joblib
    model = joblib.load('models/best_model.pkl')
try:
    scaler = joblib.load('models/scaler.pkl')
    logging.info("Loaded scaler from file.")
except Exception as e:
    logging.error(f"Failed to load scaler: {e}")
    scaler = None

def predict_risk(features: List[CustomerFeatures]) -> List[dict]:
    """
    Predict risk level probabilities for input features.
    
    Args:
        features: List of Pydantic CustomerFeatures objects.
    
    Returns:
        List of dictionaries with predicted probabilities.
    """
    feature_names = ['LogMonetary', 'Frequency', 'LogAvgTransactionAmount', 
                    'NightRatio', 'Freq_FinancialServices', 'Freq_Airtime']
    
    data = {key: [getattr(f, key) for f in features] for key in feature_names}
    df = pd.DataFrame(data)
    
    try:
        scaled_data = scaler.transform(df) if scaler else df
        probabilities = model.predict_proba(scaled_data)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
    
    risk_levels = ['Low', 'Medium', 'High']
    results = []
    for prob in probabilities:
        result = {risk_levels[i]: float(prob[i]) for i in range(len(risk_levels))}
        results.append(result)
    
    return results