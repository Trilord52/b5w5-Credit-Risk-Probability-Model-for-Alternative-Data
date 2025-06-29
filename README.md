# b5w5 Credit Risk Probability Model for Alternative Data 🇪🇹💳

Building a Credit Scoring Model for Bati Bank's Buy-Now-Pay-Later Service

## Project Overview
This project develops a **Credit Risk Probability Model** for Bati Bank, in partnership with an eCommerce platform, to enable a buy-now-pay-later service. Using alternative data (transactional and behavioral), the model predicts customer creditworthiness, assigns risk probabilities, derives credit scores, and recommends optimal loan amounts and durations. Key tasks include:

- 🧠 Creating a proxy target variable using Recency, Frequency, and Monetary (RFM) metrics.
- 🔍 Performing exploratory data analysis (EDA) to uncover patterns.
- ⚙️ Engineering features with Weight of Evidence (WoE) and Information Value (IV).
- 🤖 Training machine learning models (e.g., Logistic Regression, Gradient Boosting).
- 🚀 Deploying the model as a FastAPI service with a CI/CD pipeline.
- 📜 Ensuring compliance with Basel II Capital Accord for risk measurement.

The project leverages MLOps practices with MLflow for experiment tracking and Docker for deployment.

## Project Structure
- `data/raw/`: Raw data files (e.g., transactions, variable definitions).
- `data/processed/`: Processed datasets for model training.
- `notebooks/`: Jupyter notebooks for EDA (`1.0-eda.ipynb`).
- `src/`: Scripts for data processing, training, prediction, and API.
- `src/api/`: FastAPI application for model serving.
- `tests/`: Unit tests for data processing.
- `.github/workflows/`: CI/CD pipeline configuration.
- `Dockerfile`: Container setup for the API.
- `docker-compose.yml`: Container orchestration.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Quick Start
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/b5w5-Credit-Risk-Probability-Model-for-Alternative-Data.git
   cd b5w5-Credit-Risk-Probability-Model-for-Alternative-Data