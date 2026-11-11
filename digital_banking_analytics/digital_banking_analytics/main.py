"""Predictive Analytics for Digital Banking - end-to-end pipeline."""
import pandas as pd
from data_generator import generate_customers, generate_transactions
from models import train_churn_model, train_fraud_model


def main():
    customers = generate_customers()
    txns = generate_transactions()
    customers.to_csv("data/customers.csv", index=False)
    txns.to_csv("data/transactions.csv", index=False)
    print(f"Churn rate: {customers.churned.mean():.1%} | Fraud rate: {txns.is_fraud.mean():.2%}")

    churn_model, churn_cols = train_churn_model(customers)
    train_fraud_model(txns)

    # Score customers and list the top churn risks for retention campaigns
    customers["churn_probability"] = churn_model.predict_proba(customers[churn_cols])[:, 1]
    top = customers.sort_values("churn_probability", ascending=False).head(10)
    top[["customer_id", "churn_probability"]].to_csv("outputs/top_churn_risks.csv", index=False)
    print("\nTop 10 at-risk customers:\n", top[["customer_id", "churn_probability"]].to_string(index=False))
    print("\nDone. Charts, models and CSVs saved in ./outputs")


if __name__ == "__main__":
    main()
