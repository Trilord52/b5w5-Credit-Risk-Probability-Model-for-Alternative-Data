import joblib
import numpy as np
import pandas as pd
from typing import List
from pydantic_models import CustomerFeatures

# Load the trained model and scaler
model = joblib.load('../../models/best_model.pkl')
scaler = joblib.load('../../models/scaler.pkl')

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
    
    scaled_data = scaler.transform(df)
    probabilities = model.predict_proba(scaled_data)
    
    risk_levels = ['Low', 'Medium', 'High']
    results = []
    for prob in probabilities:
        result = {risk_levels[i]: float(prob[i]) for i in range(len(risk_levels))}
        results.append(result)
    
    return results