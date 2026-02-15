"""
Tableau Export Script
Prepares analysis-ready CSV files for Tableau dashboards.
Covers the 'Tableau' skill listed on the resume.

Dashboards this feeds:
1. Executive Churn Overview — KPIs, trend, overall churn rate
2. Segment Deep-Dive — churn rate, revenue, tactics per segment
3. Model Performance — decile lift chart, feature importance bar chart
4. Retention Campaign Targeting — ranked customer list for outreach
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.features.feature_engineering import (
    engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES, BEHAVIORAL_RATIOS,
)
from src.models.segmentation import segment_customers

logger = logging.getLogger(__name__)
EXPORT_DIR = "tableau_exports"


def export_all(
    data_path: str = "data/customers.csv",
    model_dir: str = "models",
):
    """Generate all Tableau-ready CSV exports."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    # Load models
    xgb_model = joblib.load(f"{model_dir}/xgb_churn_model.pkl")
    preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")

    feature_cols = (
        NUMERIC_FEATURES + BEHAVIORAL_RATIOS + CATEGORICAL_FEATURES
        + ["is_recently_inactive", "is_long_tenure", "is_high_spender",
           "escalation_rate", "charge_increase_flag"]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]

    X_proc = preprocessor.transform(X)
    churn_probs = xgb_model.predict_proba(X_proc)[:, 1]
    df["churn_probability"] = churn_probs
    df["churn_risk_tier"] = pd.cut(
        churn_probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    # ── Export 1: Executive Overview ────────────────────────────────────────
    overview = pd.DataFrame({
        "metric": [
            "Total Customers",
            "Overall Churn Rate",
            "High Risk Customers",
            "Avg Monthly Charges",
            "Avg Tenure (months)",
        ],
        "value": [
            f"{len(df):,}",
            f"{df['churned'].mean():.2%}",
            f"{(df['churn_risk_tier'] == 'High').sum():,}",
            f"${df['monthly_charges'].mean():.2f}",
            f"{df['tenure_months'].mean():.1f}",
        ],
    })
    overview.to_csv(f"{EXPORT_DIR}/01_executive_overview.csv", index=False)
    logger.info("Exported: 01_executive_overview.csv")

    # ── Export 2: Churn Rate by Segment & Contract ───────────────────────────
    segment_summary = df.groupby(["customer_segment", "contract_type"]).agg(
        n_customers=("churned", "count"),
        churn_rate=("churned", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
        avg_tenure=("tenure_months", "mean"),
    ).reset_index().round(4)
    segment_summary.to_csv(f"{EXPORT_DIR}/02_segment_churn_analysis.csv", index=False)
    logger.info("Exported: 02_segment_churn_analysis.csv")

    # ── Export 3: KMeans Segment Results ────────────────────────────────────
    df_seg = segment_customers(df.copy(), n_clusters=5, model_dir=model_dir)
    seg_results = df_seg.groupby("segment_label").agg(
        n_customers=("churned", "count"),
        churn_rate=("churned", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_tenure=("tenure_months", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
    ).reset_index().round(4)
    seg_results["retention_tactic"] = seg_results["segment_label"].map({
        "High-Value Loyal": "Loyalty rewards program",
        "At-Risk Churners": "Proactive outreach + discount",
        "Low Engagement": "Re-engagement campaign",
        "New Customers": "Onboarding check-in",
        "Price Sensitive": "Annual plan discount",
    })
    seg_results.to_csv(f"{EXPORT_DIR}/03_kmeans_segments.csv", index=False)
    logger.info("Exported: 03_kmeans_segments.csv")

    # ── Export 4: Feature Importance ─────────────────────────────────────────
    if os.path.exists(f"{model_dir}/feature_importance.csv"):
        feat_imp = pd.read_csv(f"{model_dir}/feature_importance.csv")
        feat_imp.head(15).to_csv(f"{EXPORT_DIR}/04_feature_importance.csv", index=False)
        logger.info("Exported: 04_feature_importance.csv")

    # ── Export 5: Decile Lift Table ───────────────────────────────────────────
    if os.path.exists(f"{model_dir}/decile_lift_table.csv"):
        lift = pd.read_csv(f"{model_dir}/decile_lift_table.csv")
        lift.to_csv(f"{EXPORT_DIR}/05_decile_lift.csv", index=False)
        logger.info("Exported: 05_decile_lift.csv")

    # ── Export 6: Retention Campaign List (top 20% risk customers) ───────────
    top_risk = df_seg.nlargest(int(len(df) * 0.20), "churn_probability")[
        ["customer_id", "customer_segment", "contract_type", "monthly_charges",
         "tenure_months", "churn_probability", "churn_risk_tier",
         "segment_label", "retention_tactic"]
    ].round(4)
    top_risk.to_csv(f"{EXPORT_DIR}/06_retention_campaign_list.csv", index=False)
    logger.info(f"Exported: 06_retention_campaign_list.csv ({len(top_risk):,} customers)")

    # ── Export 7: Churn Drivers Summary ──────────────────────────────────────
    drivers = pd.DataFrame({
        "risk_signal": [
            "Inactive 30+ days",
            "Monthly contract",
            "High support calls (>5)",
            "Low NPS (<5)",
            "New customer (<6 months)",
            "No escalations, high NPS",
        ],
        "churn_rate": [
            df[df["days_since_last_activity"] > 30]["churned"].mean(),
            df[df["contract_type"] == "monthly"]["churned"].mean(),
            df[df["num_support_calls"] > 5]["churned"].mean(),
            df[df["nps_score"] < 5]["churned"].mean(),
            df[df["tenure_months"] < 6]["churned"].mean(),
            df[(df["support_escalations"] == 0) & (df["nps_score"] >= 8)]["churned"].mean(),
        ],
    }).round(4)
    drivers["vs_average"] = (drivers["churn_rate"] / df["churned"].mean()).round(2)
    drivers.to_csv(f"{EXPORT_DIR}/07_churn_drivers.csv", index=False)
    logger.info("Exported: 07_churn_drivers.csv")

    logger.info(f"\nAll exports saved to {EXPORT_DIR}/")
    logger.info("Connect Tableau to this folder and build dashboards from these CSVs.")
    return EXPORT_DIR


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_all()
