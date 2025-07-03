from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    LogMonetary: float
    Frequency: float
    LogAvgTransactionAmount: float
    NightRatio: float
    Freq_FinancialServices: float
    Freq_Airtime: float

class PredictionResponse(BaseModel):
    Low: float
    Medium: float
    High: float