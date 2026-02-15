"""
Generate synthetic customer dataset (1M rows) for demo/testing purposes.
Simulates realistic churn patterns and behavioral signals.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)


def generate_customers(n: int = 1_000_000, output_path: str = "data/customers.csv"):
    """Generate synthetic customer dataset with realistic churn patterns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Base features
    tenure = np.random.exponential(scale=24, size=n).clip(1, 120).astype(int)
    contract_type = np.random.choice(
        ["monthly", "annual", "two_year"], size=n, p=[0.45, 0.35, 0.20]
    )
    segment = np.random.choice(
        ["enterprise", "smb", "consumer"], size=n, p=[0.15, 0.35, 0.50]
    )

    monthly_charges = np.where(
        segment == "enterprise", np.random.normal(800, 200, n),
        np.where(segment == "smb", np.random.normal(200, 60, n),
                 np.random.normal(60, 20, n))
    ).clip(10, 5000)

    total_charges = (monthly_charges * tenure * np.random.uniform(0.9, 1.1, n)).clip(0)

    support_calls = np.random.poisson(
        lam=np.where(contract_type == "monthly", 3.5, 1.5), size=n
    )
    escalations = (support_calls * np.random.uniform(0.0, 0.3, n)).astype(int)

    last_activity = np.random.exponential(scale=10, size=n).clip(0, 180).astype(int)
    logins_30d = np.random.poisson(
        lam=np.where(last_activity > 30, 1, 15), size=n
    ).clip(0, 60)

    transactions_90d = np.random.poisson(lam=5, size=n).clip(0)
    spend_90d = (transactions_90d * monthly_charges / 4 * np.random.uniform(0.5, 1.5, n)).clip(0)

    nps = np.random.normal(
        loc=np.where(support_calls > 5, 4, 7.5), scale=1.5, size=n
    ).clip(0, 10)

    contract_remaining = np.where(
        contract_type == "monthly", 0,
        np.where(contract_type == "annual",
                 np.random.randint(0, 12, n),
                 np.random.randint(0, 24, n))
    )

    payment = np.random.choice(
        ["credit_card", "bank_transfer", "electronic_check"], size=n, p=[0.4, 0.35, 0.25]
    )
    internet = np.random.choice(["fiber", "DSL", "none"], size=n, p=[0.5, 0.35, 0.15])
    region = np.random.choice(["northeast", "southeast", "midwest", "west", "south"], size=n)
    channel = np.random.choice(["organic", "paid", "referral"], size=n, p=[0.4, 0.35, 0.25])

    # Churn probability — higher for monthly, high support, low engagement
    churn_prob = (
        0.05
        + 0.15 * (contract_type == "monthly")
        + 0.10 * (support_calls > 5)
        + 0.12 * (last_activity > 30)
        + 0.08 * (nps < 5)
        + 0.06 * (logins_30d < 3)
        - 0.08 * (tenure > 24)
        - 0.05 * (contract_type == "two_year")
        + np.random.normal(0, 0.03, n)
    ).clip(0, 1)

    churned = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:08d}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "num_products": np.random.randint(1, 6, n),
        "num_support_calls": support_calls,
        "support_escalations": escalations,
        "days_since_last_activity": last_activity,
        "avg_session_duration_mins": np.random.exponential(15, n).clip(1, 120).round(1),
        "num_logins_30d": logins_30d,
        "num_transactions_90d": transactions_90d,
        "total_spend_90d": spend_90d.round(2),
        "nps_score": nps.round(1),
        "contract_months_remaining": contract_remaining,
        "contract_type": contract_type,
        "payment_method": payment,
        "internet_service": internet,
        "customer_segment": segment,
        "region": region,
        "acquisition_channel": channel,
        "churned": churned,
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {n:,} customer records → {output_path}")
    print(f"Churn rate: {churned.mean():.2%}")
    return df


if __name__ == "__main__":
    generate_customers(n=1_000_000)
