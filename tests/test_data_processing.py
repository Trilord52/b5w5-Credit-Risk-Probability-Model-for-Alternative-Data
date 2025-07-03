import pytest
import pandas as pd
from src.data_processing import load_data
from src.feature_engineering import create_rfm_features

def test_load_data(tmp_path):
    # Create a sample CSV
    d = tmp_path / "sample.csv"
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df.to_csv(d, index=False)
    loaded = load_data(str(d))
    assert loaded.shape == (2, 2)

def test_create_rfm_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionId': [10, 11, 12],
        'Amount': [100, 200, 300],
        'TransactionDate': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    })
    out = create_rfm_features(df)
    assert 'Recency' in out.columns
    assert 'Frequency' in out.columns
    assert 'Monetary' in out.columns
