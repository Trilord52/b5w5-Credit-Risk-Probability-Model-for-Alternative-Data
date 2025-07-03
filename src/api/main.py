from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerFeatures, PredictionResponse  # Correct relative import
from . import model  # Relative import from the same directory
from typing import List

app = FastAPI(title="Credit Risk Prediction API")

@app.post("/predict/", response_model=List[PredictionResponse])
async def predict_risk(features: List[CustomerFeatures]):
    """
    Endpoint to predict risk level probabilities for customer features.
    """
    try:
        predictions = model.predict_risk(features)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)