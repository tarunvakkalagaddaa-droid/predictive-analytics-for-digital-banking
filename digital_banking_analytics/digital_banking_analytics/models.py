"""Train and evaluate churn and fraud prediction models."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split


def _evaluate(name, model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    print(f"\n===== {name} (threshold={threshold}) =====")
    print(classification_report(y_test, pred, digits=3))
    auc = roc_auc_score(y_test, proba)
    print(f"ROC-AUC: {auc:.3f}")
    return auc


def _plot_importance(model, cols, title, path):
    imp = pd.Series(model.feature_importances_, index=cols).sort_values()
    imp.plot.barh(figsize=(7, 4), title=title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def train_churn_model(customers):
    X = customers.drop(columns=["customer_id", "churned"])
    y = customers["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    _evaluate("Customer Churn (Random Forest)", model, X_te, y_te, threshold=0.5)
    _plot_importance(model, X.columns, "Churn - Feature Importance", "outputs/churn_importance.png")
    RocCurveDisplay.from_estimator(model, X_te, y_te)
    plt.savefig("outputs/churn_roc.png", dpi=120); plt.close()
    joblib.dump((model, list(X.columns)), "outputs/churn_model.joblib")
    return model, X.columns


def train_fraud_model(txns):
    X = pd.get_dummies(txns.drop(columns=["txn_id", "customer_id", "is_fraud"]), columns=["channel"])
    y = txns["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = GradientBoostingClassifier(random_state=42)
    weights = y_tr.map({0: 1.0, 1: (y_tr == 0).sum() / (y_tr == 1).sum()})  # handle imbalance
    model.fit(X_tr, y_tr, sample_weight=weights)
    _evaluate("Transaction Fraud (Gradient Boosting)", model, X_te, y_te, threshold=0.5)
    _plot_importance(model, X.columns, "Fraud - Feature Importance", "outputs/fraud_importance.png")
    RocCurveDisplay.from_estimator(model, X_te, y_te)
    plt.savefig("outputs/fraud_roc.png", dpi=120); plt.close()
    joblib.dump((model, list(X.columns)), "outputs/fraud_model.joblib")
    return model, X.columns
