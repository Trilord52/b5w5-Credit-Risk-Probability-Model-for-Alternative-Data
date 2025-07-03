import pandas as pd
import numpy as np
import os
from src.train import train_and_log_model

def test_train_and_log_model(tmp_path):
    # Create a small synthetic dataset
    df = pd.DataFrame({
        'LogMonetary': np.random.rand(20),
        'Frequency': np.random.randint(1, 10, 20),
        'LogAvgTransactionAmount': np.random.rand(20),
        'NightRatio': np.random.rand(20),
        'Freq_FinancialServices': np.random.randint(0, 5, 20),
        'Freq_Airtime': np.random.randint(0, 5, 20),
        'RiskCluster': np.random.choice([0, 1, 2], 20)
    })
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    df.to_csv(data_path, index=False)
    model, scaler = train_and_log_model(str(data_path), str(model_path), str(scaler_path))
    assert os.path.exists(model_path)
    assert os.path.exists(scaler_path)
    # Check model and scaler are usable
    X = df[['LogMonetary', 'Frequency', 'LogAvgTransactionAmount', 'NightRatio', 'Freq_FinancialServices', 'Freq_Airtime']]
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)
    assert proba.shape[0] == 20 