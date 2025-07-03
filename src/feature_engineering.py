import pandas as pd
import numpy as np
import logging

def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Recency, Frequency, and Monetary features."""
    # Example logic, replace with actual from notebook
    df['Recency'] = (df['TransactionDate'].max() - df['TransactionDate']).dt.days
    df['Frequency'] = df.groupby('CustomerId')['TransactionId'].transform('count')
    df['Monetary'] = df.groupby('CustomerId')['Amount'].transform('sum')
    return df

# Add more feature engineering functions as needed 