import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize
import logging

def train_and_log_model(data_path: str, model_out_path: str, scaler_out_path: str):
    """Train models, tune hyperparameters, log with MLflow, and save best model and scaler."""
    df = pd.read_csv(data_path)
    features = ['LogMonetary', 'Frequency', 'LogAvgTransactionAmount', 'NightRatio', 'Freq_FinancialServices', 'Freq_Airtime']
    y = df['RiskCluster']
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, scaler_out_path)
    logging.info(f"Scaler saved to {scaler_out_path}")
    # Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr_params = {'C': [0.1, 1, 10]}
    lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='f1_weighted')
    lr_grid.fit(X_train_scaled, y_train)
    lr_best = lr_grid.best_estimator_
    # Gradient Boosting
    gb = GradientBoostingClassifier()
    gb_params = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='f1_weighted')
    gb_grid.fit(X_train_scaled, y_train)
    gb_best = gb_grid.best_estimator_
    # Evaluate
    models = {'LogisticRegression': lr_best, 'GradientBoosting': gb_best}
    best_model = None
    best_f1 = -1
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        # ROC-AUC (multiclass)
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        try:
            roc_auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
        except Exception as e:
            roc_auc = float('nan')
            logging.warning(f"ROC-AUC calculation failed: {e}")
        mlflow.sklearn.log_model(model, name)
        mlflow.log_metric(f"{name}_f1", f1)
        mlflow.log_metric(f"{name}_accuracy", acc)
        mlflow.log_metric(f"{name}_precision", prec)
        mlflow.log_metric(f"{name}_recall", rec)
        mlflow.log_metric(f"{name}_roc_auc", roc_auc)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    joblib.dump(best_model, model_out_path)
    logging.info(f"Best model saved to {model_out_path}")
    return best_model, scaler
