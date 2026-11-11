# Predictive Analytics for Digital Banking

A simple base project that uses machine learning on (synthetic) digital-banking data.

| Use case | Model | Target |
|---|---|---|
| Customer churn prediction | Random Forest | `churned` |
| Transaction fraud detection | Gradient Boosting | `is_fraud` |

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Files
- `data_generator.py` – creates synthetic customers and transactions (`data/`)
- `models.py` – training, evaluation (precision/recall/ROC-AUC), feature-importance plots
- `main.py` – runs the full pipeline, saves models, charts and the top churn-risk list to `outputs/`

## Next steps
Swap the synthetic CSVs for real data, add SMOTE / class weights for fraud imbalance, tune hyperparameters, add a Streamlit dashboard.
