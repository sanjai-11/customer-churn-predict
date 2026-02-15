-- ============================================================
-- Customer Churn Prediction — SQL Queries
-- These queries represent the data extraction and feature
-- aggregation work done before Python modeling.
-- Compatible with PostgreSQL / BigQuery / Snowflake
-- ============================================================


-- ── 1. Load raw customer base with activity summary ──────────────────────────
-- Used to pull the 1M+ customer dataset from a data warehouse

CREATE OR REPLACE VIEW churn_feature_base AS
SELECT
    c.customer_id,
    c.acquisition_channel,
    c.contract_type,
    c.payment_method,
    c.internet_service,
    c.customer_segment,
    c.region,

    -- Tenure
    DATEDIFF('month', c.created_at, CURRENT_DATE)          AS tenure_months,

    -- Billing
    b.monthly_charges,
    b.total_charges,
    b.contract_months_remaining,

    -- Support behaviour
    COUNT(s.ticket_id)                                      AS num_support_calls,
    COUNT(CASE WHEN s.escalated = TRUE THEN 1 END)          AS support_escalations,

    -- Engagement (last 30 days)
    COUNT(DISTINCT l.session_id)                            AS num_logins_30d,
    AVG(l.session_duration_mins)                            AS avg_session_duration_mins,
    MAX(l.created_at)                                       AS last_activity_at,
    DATEDIFF('day', MAX(l.created_at), CURRENT_DATE)        AS days_since_last_activity,

    -- Transactions (last 90 days)
    COUNT(t.transaction_id)                                 AS num_transactions_90d,
    SUM(t.amount)                                           AS total_spend_90d,

    -- NPS
    AVG(n.score)                                            AS nps_score,

    -- Target label
    CASE WHEN c.churned_at IS NOT NULL
         AND c.churned_at <= CURRENT_DATE THEN 1 ELSE 0 END AS churned

FROM customers c
LEFT JOIN billing b           ON c.customer_id = b.customer_id
LEFT JOIN support_tickets s   ON c.customer_id = s.customer_id
LEFT JOIN login_sessions l    ON c.customer_id = l.customer_id
    AND l.created_at >= CURRENT_DATE - INTERVAL '30 days'
LEFT JOIN transactions t      ON c.customer_id = t.customer_id
    AND t.created_at >= CURRENT_DATE - INTERVAL '90 days'
LEFT JOIN nps_responses n     ON c.customer_id = n.customer_id

GROUP BY
    c.customer_id, c.acquisition_channel, c.contract_type,
    c.payment_method, c.internet_service, c.customer_segment,
    c.region, c.created_at, c.churned_at,
    b.monthly_charges, b.total_charges, b.contract_months_remaining;


-- ── 2. Churn rate by segment and contract type ───────────────────────────────
-- Key business insight query — presented to stakeholders

SELECT
    customer_segment,
    contract_type,
    COUNT(*)                                    AS total_customers,
    SUM(churned)                                AS churned_count,
    ROUND(AVG(churned) * 100, 2)               AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)             AS avg_monthly_charges,
    ROUND(AVG(tenure_months), 1)               AS avg_tenure_months
FROM churn_feature_base
GROUP BY customer_segment, contract_type
ORDER BY churn_rate_pct DESC;


-- ── 3. Top behavioral churn signals ─────────────────────────────────────────
-- Validates feature importance output from XGBoost

SELECT
    CASE
        WHEN days_since_last_activity > 30  THEN 'Inactive 30+ days'
        WHEN num_support_calls > 5          THEN 'High support calls'
        WHEN nps_score < 5                  THEN 'Low NPS (<5)'
        WHEN contract_type = 'monthly'      THEN 'Monthly contract'
        WHEN tenure_months < 6              THEN 'New customer (<6mo)'
        ELSE 'Other'
    END                                         AS risk_signal,
    COUNT(*)                                    AS n_customers,
    ROUND(AVG(churned) * 100, 2)               AS churn_rate_pct
FROM churn_feature_base
GROUP BY 1
ORDER BY churn_rate_pct DESC;


-- ── 4. Segment summary after KMeans (joined from Python output) ──────────────
-- After running segmentation.py, load results back into warehouse

CREATE TABLE IF NOT EXISTS customer_segments (
    customer_id     VARCHAR(50) PRIMARY KEY,
    segment_id      INT,
    segment_label   VARCHAR(50),
    retention_tactic TEXT,
    scored_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query: churn rate and avg charges per segment
SELECT
    cs.segment_label,
    COUNT(*)                                    AS n_customers,
    ROUND(AVG(cf.churned) * 100, 2)            AS churn_rate_pct,
    ROUND(AVG(cf.monthly_charges), 2)          AS avg_monthly_charges,
    ROUND(AVG(cf.tenure_months), 1)            AS avg_tenure_months,
    cs.retention_tactic
FROM customer_segments cs
JOIN churn_feature_base cf ON cs.customer_id = cf.customer_id
GROUP BY cs.segment_label, cs.retention_tactic
ORDER BY churn_rate_pct DESC;


-- ── 5. Decile lift validation query ─────────────────────────────────────────
-- Validates the 30% retention lift claim in SQL

WITH scored AS (
    SELECT
        cf.customer_id,
        cf.churned,
        cp.churn_probability,
        NTILE(10) OVER (ORDER BY cp.churn_probability DESC) AS decile
    FROM churn_feature_base cf
    JOIN churn_predictions cp ON cf.customer_id = cp.customer_id
),
overall AS (
    SELECT AVG(churned) AS base_rate FROM churn_feature_base
)
SELECT
    s.decile,
    COUNT(*)                                            AS n_customers,
    SUM(s.churned)                                      AS n_churners,
    ROUND(AVG(s.churned) * 100, 2)                     AS churn_rate_pct,
    ROUND(AVG(s.churned) / o.base_rate, 2)             AS lift
FROM scored s, overall o
GROUP BY s.decile, o.base_rate
ORDER BY s.decile;


-- ── 6. Monthly churn trend ───────────────────────────────────────────────────
-- Used for Tableau time-series dashboard

SELECT
    DATE_TRUNC('month', churned_at)             AS churn_month,
    customer_segment,
    contract_type,
    COUNT(*)                                    AS churned_customers,
    SUM(monthly_charges)                        AS lost_mrr
FROM customers
WHERE churned_at IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1 DESC;
