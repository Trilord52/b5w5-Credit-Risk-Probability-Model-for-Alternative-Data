import pandas as pd
import logging

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

# Add more preprocessing functions as needed