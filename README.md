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

## Methods

## Task1: Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord, as outlined in the *Credit Risk Analysis and Modeling* presentation, emphasizes robust risk measurement to ensure financial institutions maintain adequate capital reserves against potential losses. This regulatory framework requires transparent and auditable credit risk models to comply with standards for risk assessment and capital allocation. Interpretable models, such as Logistic Regression with Weight of Evidence (WoE), provide clear coefficients that explain how each feature (e.g., transaction frequency) contributes to the risk prediction, making it easier to justify decisions to regulators. Well-documented models, with detailed feature engineering and evaluation processes, ensure traceability and compliance, reducing the risk of regulatory penalties and building trust with stakeholders for Bati Bank’s buy-now-pay-later service.

### Why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Since the dataset lacks a direct "default" label, as noted in the *Credit Risk Analysis and Modeling* presentation’s discussion on alternative data, a proxy variable is necessary to approximate credit risk. Using Recency, Frequency, and Monetary (RFM) metrics, we can identify disengaged customers (e.g., those with low transaction frequency and monetary value) as high-risk proxies for potential default. However, this approach carries business risks, including misclassification if the proxy does not accurately reflect true default behavior. For example, a customer with low engagement may not necessarily default, leading to false positives that deny credit to viable borrowers, or false negatives that approve risky loans, potentially causing financial losses or reputational damage for Bati Bank.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
As highlighted in the *Credit Risk Analysis and Modeling* presentation, simple models like Logistic Regression with WoE are highly interpretable, with coefficients that clearly indicate feature impact, making them regulator-friendly and easy to audit in a Basel II-compliant environment. They are computationally efficient but may miss complex, non-linear patterns in alternative data, potentially reducing predictive accuracy. Conversely, Gradient Boosting Machines (GBMs) capture non-linear relationships for higher accuracy, as noted in the presentation, but are less interpretable, posing challenges for regulatory scrutiny and requiring Explainable AI (XAI) techniques. In a regulated financial context, interpretability often outweighs marginal accuracy gains to ensure compliance, transparency, and trust, though GBMs may be preferred if accuracy is critical and interpretability can be addressed through additional tools.

### Task 2: Exploratory Data Analysis

In Task 2, I conducted a comprehensive EDA on the Xente transaction dataset (~96,000 rows) using `notebooks/1.0-eda.ipynb`. The analysis, guided by *Xente Variable Definitions* and the *Credit Risk Analysis and Modeling* presentation, used **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**. Key steps included:
- **Data Loading**: Loaded `data.csv` and `xente_variable_definitions.csv` to understand structure (19 columns, 2,094 unique `CustomerId`).
- **Summary Statistics**: Computed statistics for numerical features (`Amount`: mean=6,717.8, std=123,306.8) and categorical features (e.g., `financial_services`=45,405, `airtime`=45,027).
- **Visualizations**: Generated color-coded plots (saved in `plots/`):
  - **Amount Distribution** (`amount_dist.png`): Histogram (blue) showing skewness and outliers.
  - **Value Distribution** (`value_dist.png`): Histogram (blue) of absolute transaction values.
  - **Product Category Counts** (`product_category_counts.png`): Count plot (green) highlighting dominant categories.
  - **Transaction Hour Distribution** (`transaction_hour_dist.png`): Count plot (red) showing peak activity times.
  - **Correlation Heatmap** (`correlation_heatmap.png`): Heatmap (purple) of numerical and temporal features.
- **Missing Values**: No missing values detected, simplifying preprocessing; proposed imputation for future data (e.g., `CountryCode` with mode=256).
- **Outlier Detection**: Identified outliers in `Amount` (range: -1,000,000 to 9,880,000) using box plots, suggesting capping at 1.5*IQR or log transformation.
- **Insights**:
  1. Dominant categories (`financial_services`, `airtime`) are key for RFM analysis.
  2. High `Amount` variability requires outlier handling.
  3. Temporal features (`TransactionHour`) reveal behavioral patterns.
  4. `FraudResult` imbalance (193 positives) suggests limited use as a proxy.
  5. No missing values simplifies preprocessing; future data may need imputation.

The notebook and plots were committed to the `task-2` branch with a pull request, informing feature engineering (Task 3).

### Task 3: Feature Engineering

Completed in `notebooks/2.0-feature-engineering.ipynb`:
- **RFM Features**: Recency (days since last transaction), Frequency (transaction count), Monetary (total spend, log-transformed).
- **Behavioral Features**: Average transaction amount (log-transformed), counts for `financial_services` and `airtime`, night activity ratio (0–6 AM).
- **Outlier Handling**: Log transformation for `Monetary` and `AvgTransactionAmount` to reduce variance.
- **WoE/IV Analysis**: Used `scorecardpy` to rank features by predictive power with `FraudResult` as temporary target. Replaced `woebin_plot` with custom plotting to avoid `TypeError`.
- **Output**: Saved features to `data/processed/customer_features.csv`.
- **Insights**: High-IV features (e.g., `LogMonetary`, `Frequency`) will guide proxy creation in Task 4.
- **Fixes**: 
  - Fixed `ImportError` by pinning `scipy==1.15.3`, `statsmodels==0.14.4`, `scorecardpy==0.1.9.2`.
  - Fixed `TypeError: unhashable type: 'numpy.ndarray'` in `woebin_plot` by using custom plotting in cell 5.
  - Fixed `KeyError: 'FraudResult'` with `how='left'` and `fillna(0)` in cell 5.
  - Fixed `MergeError` by dropping duplicate `FraudResult` column in cell 5.
  - Fixed `TypeError` in `woebin_ply` by converting `variable` column to strings in `woe_bins_copy` in cell 6.
Committed to `task-3` branch with pull request,


### Task 4: Proxy Variable Creation

Completed in `notebooks/3.0-proxy-variable.ipynb`:
- Loaded features from `data/processed/customer_features.csv`.
- Selected high-IV features (e.g., `LogMonetary_woe`, `Frequency_woe`) based on Task 3 IV scores.
- Applied K-means clustering (3–5 clusters), selected best using silhouette score.
- Created `RiskCluster` proxy and mapped to `RiskLevel` (low/medium/high) based on `FraudResult` means.
- Saved features with proxy to `data/processed/customer_features_with_proxy.csv`.
- **Insights**: Clusters reflect risk patterns; high-IV features drive segmentation. Ready for Task 5 modeling.
Committed to `task-4` branch with pull request.

### Task 5: Model Training and Evaluation

Completed in `notebooks/4.0-model-training.ipynb`:
- Loaded features from `data/processed/customer_features_with_proxy.csv`.
- Trained Logistic Regression and Gradient Boosting on `RiskCluster` using high-IV features.
- Evaluated models with accuracy, precision, recall, and F1-score; visualized confusion matrices.
- Selected best model based on F1-score and saved to `models/best_model.pkl`.
- **Insights**: Logistic Regression ensures interpretability; Gradient Boosting may improve performance. Ready for Task 6 deployment.
Committed to `task-5` branch with pull request.

## Project/Repository Structure

- `data/raw/`: Raw data files (`transactions.csv`, `xente_variable_definitions.csv`).
- `data/processed/`: Processed datasets (to be populated).
- `notebooks/`: Jupyter Notebooks for EDA and analysis (`1.0-eda.ipynb`, `2.0-feature-engineering.ipynb`).
- `src/`: Python scripts for data processing, training, and prediction.
- `src/api/`: FastAPI application files.
- `tests/`: Unit tests for scripts.
- `plots/`: Exported visualizations from EDA.
- `requirements.txt`: Project dependencies.
- `Dockerfile`: Docker configuration for deployment.
- `docker-compose.yml`: Docker Compose setup.
- `.github/workflows/ci.yml`: CI/CD pipeline configuration.

## Quick Start
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Trilord52/b5w5-Credit-Risk-Probability-Model-for-Alternative-Data.git
   cd b5w5-Credit-Risk-Probability-Model-for-Alternative-Data
2. **Create and activate a virtual environment:**:
    ```bash
    python -m venv env
    env\Scripts\activate      # Windows
3. **Install dependencies:**:
    ```bash
    pip install -r requirements.txt
4. **Run the EDA notebook:**:
    ```bash
    jupyter notebook notebooks/1.0-eda.ipynb
