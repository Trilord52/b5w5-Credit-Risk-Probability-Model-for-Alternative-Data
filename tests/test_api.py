from fastapi.testclient import TestClient
from src.api.main import app

def test_predict_endpoint():
    client = TestClient(app)
    payload = [
        {
            "LogMonetary": 10.0,
            "Frequency": 5,
            "LogAvgTransactionAmount": 2.5,
            "NightRatio": 0.1,
            "Freq_FinancialServices": 1,
            "Freq_Airtime": 2
        }
    ]
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert all(isinstance(item, dict) for item in response.json()) 