"""
Feature engineering pipeline for customer churn prediction.
Handles 1M+ customer records with behavioral feature extraction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


# ─── Feature definitions ──────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "num_support_calls",
    "days_since_last_activity",
    "avg_session_duration_mins",
    "num_logins_30d",
    "num_transactions_90d",
    "total_spend_90d",
    "support_escalations",
    "nps_score",
    "contract_months_remaining",
]

CATEGORICAL_FEATURES = [
    "contract_type",       # monthly, annual, two-year
    "payment_method",      # credit_card, bank_transfer, electronic_check
    "internet_service",    # DSL, fiber, none
    "customer_segment",    # enterprise, smb, consumer
    "region",
    "acquisition_channel", # organic, paid, referral
]

BEHAVIORAL_RATIOS = [
    "support_rate",        # support_calls / tenure
    "spend_per_month",     # total_charges / tenure_months
    "engagement_score",    # sessions * avg_duration / 30
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate behavioral and ratio features from raw customer data.
    Handles 1M+ rows efficiently using vectorized operations.
    """
    df = df.copy()

    # Prevent division by zero
    df["tenure_months"] = df["tenure_months"].clip(lower=1)

    # Behavioral ratios
    df["support_rate"] = df["num_support_calls"] / df["tenure_months"]
    df["spend_per_month"] = df["total_charges"] / df["tenure_months"]
    df["engagement_score"] = (
        df["num_logins_30d"] * df.get("avg_session_duration_mins", 1)
    ) / 30.0

    # Recency-based flags
    df["is_recently_inactive"] = (df["days_since_last_activity"] > 30).astype(int)
    df["is_long_tenure"] = (df["tenure_months"] > 24).astype(int)
    df["is_high_spender"] = (
        df["monthly_charges"] > df["monthly_charges"].quantile(0.75)
    ).astype(int)

    # Interaction features
    df["escalation_rate"] = df.get("support_escalations", 0) / df["num_support_calls"].clip(lower=1)
    df["charge_increase_flag"] = (
        df["monthly_charges"] > df["total_charges"] / df["tenure_months"]
    ).astype(int)

    logger.info(f"Engineered features for {len(df):,} records. Shape: {df.shape}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", LabelEncoder()),
    ])

    # Use OrdinalEncoder for pipeline compatibility
    from sklearn.preprocessing import OrdinalEncoder
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    all_numeric = NUMERIC_FEATURES + BEHAVIORAL_RATIOS + [
        "is_recently_inactive", "is_long_tenure", "is_high_spender",
        "escalation_rate", "charge_increase_flag",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, [f for f in all_numeric if f in all_numeric]),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_and_prepare(path: str, chunksize: int = 100_000) -> pd.DataFrame:
    """
    Load large CSV in chunks (supports 1M+ rows) and apply feature engineering.
    """
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunks.append(engineer_features(chunk))
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded and prepared {len(df):,} total records")
    return df
