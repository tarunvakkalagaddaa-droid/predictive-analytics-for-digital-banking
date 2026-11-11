"""Generates synthetic digital-banking data (customers + transactions)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def generate_customers(n=5000):
    df = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": rng.integers(18, 75, n),
        "tenure_months": rng.integers(1, 120, n),
        "monthly_logins": rng.poisson(12, n),
        "avg_balance": rng.gamma(2.0, 2500, n).round(2),
        "num_products": rng.integers(1, 6, n),
        "monthly_txn_count": rng.poisson(25, n),
        "support_tickets": rng.poisson(1.2, n),
        "uses_mobile_app": rng.integers(0, 2, n),
        "credit_score": rng.normal(680, 70, n).clip(300, 850).round(),
    })
    # Churn is more likely with low engagement, many complaints, few products
    score = (
        -0.12 * df.monthly_logins
        + 0.45 * df.support_tickets
        - 0.35 * df.num_products
        - 0.015 * df.tenure_months
        - 0.5 * df.uses_mobile_app
        - 0.00005 * df.avg_balance
        + 1.5
        + rng.normal(0, 0.7, n)
    )
    df["churned"] = (1 / (1 + np.exp(-score)) > 0.5).astype(int)
    return df


def generate_transactions(n=30000, n_customers=5000):
    df = pd.DataFrame({
        "txn_id": np.arange(1, n + 1),
        "customer_id": rng.integers(1, n_customers + 1, n),
        "amount": rng.lognormal(4.0, 1.0, n).round(2),
        "hour": rng.integers(0, 24, n),
        "is_international": (rng.random(n) < 0.05).astype(int),
        "is_new_device": (rng.random(n) < 0.08).astype(int),
        "txn_last_24h": rng.poisson(2, n),
        "channel": rng.choice(["mobile", "web", "atm", "pos"], n, p=[.45, .25, .1, .2]),
    })
    risk = (
        0.0015 * df.amount
        + 1.6 * df.is_international
        + 1.4 * df.is_new_device
        + 0.25 * df.txn_last_24h
        + 0.9 * df.hour.isin([0, 1, 2, 3, 4]).astype(int)
        - 6.0
        + rng.normal(0, 0.6, n)
    )
    df["is_fraud"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    return df


if __name__ == "__main__":
    c, t = generate_customers(), generate_transactions()
    c.to_csv("data/customers.csv", index=False)
    t.to_csv("data/transactions.csv", index=False)
    print(f"Customers: {c.shape}, churn rate {c.churned.mean():.1%}")
    print(f"Transactions: {t.shape}, fraud rate {t.is_fraud.mean():.2%}")
