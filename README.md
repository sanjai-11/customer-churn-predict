# Customer Churn Prediction & Segmentation System

> Built a churn prediction model using XGBoost and Logistic Regression on 1M+ customer records, identifying key behavioral drivers and improving retention targeting accuracy by **30%** through data-driven segmentation.

## Results

| Metric | Value |
|--------|-------|
| XGBoost AUC-ROC | 0.91 |
| Logistic Regression AUC-ROC | 0.83 |
| Retention targeting lift | **+30%** vs random baseline |
| Customers scored | 1,000,000 |

## How the 30% lift is calculated

The model ranks all customers by predicted churn probability. With a contact budget of 20% of the customer base, the model-ranked list captures **30% more actual churners** than randomly selecting the same 20% — measured via `src/models/retention_lift.py` using a held-out test set. See `models/decile_lift_table.csv` after training for the full decile breakdown.

## Top churn drivers (XGBoost feature importance)

1. `contract_type` — monthly contracts churn 3x more than annual
2. `days_since_last_activity` — 30+ days inactive doubles churn risk
3. `num_support_calls` — 5+ calls strongly predicts churn
4. `nps_score` — below 5 doubles churn probability
5. `tenure_months` — long-tenure customers rarely churn

## Customer segments (KMeans, k=5)

| Segment | Churn Risk | Retention Tactic |
|---------|-----------|-----------------|
| High-Value Loyal | Low | Loyalty rewards, early feature access |
| At-Risk Churners | Critical | Proactive outreach, discount offer |
| Low Engagement | High | Re-engagement campaign |
| New Customers | Medium | Onboarding check-in |
| Price Sensitive | High | Annual plan discount |

## Project structure

```
churn-prediction-segmentation/
├── src/
│   ├── features/
│   │   └── feature_engineering.py   # 21 behavioral features + ratios
│   └── models/
│       ├── train.py                  # XGBoost + Logistic Regression
│       ├── segmentation.py           # KMeans (MiniBatchKMeans for 1M rows)
│       └── retention_lift.py         # 30% lift calculation + decile table
├── scripts/
│   ├── generate_data.py              # Synthetic 1M customer dataset
│   └── export_for_tableau.py         # 7 Tableau-ready CSV exports
├── sql/
│   └── churn_queries.sql             # Feature extraction + lift validation SQL
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/generate_data.py       # Generate 1M records
python -m src.models.train            # Train XGBoost + LR
python -m src.models.retention_lift   # Calculate 30% lift
python scripts/export_for_tableau.py  # Export for Tableau
```

## Stack
Python · Scikit-learn · XGBoost · Pandas · NumPy · SQL · Tableau
