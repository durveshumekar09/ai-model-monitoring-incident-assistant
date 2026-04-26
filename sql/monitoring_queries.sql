-- Batch volume by data source
SELECT
    data_source,
    COUNT(*) AS total_rows
FROM transactions
GROUP BY data_source;


-- Batch-level fraud rate and risk profile
SELECT
    batch_file_name,
    batch_type,
    COUNT(*) AS total_transactions,
    ROUND(AVG(label_fraud), 4) AS fraud_rate,
    ROUND(AVG(fraud_probability_simulated), 4) AS avg_simulated_fraud_probability,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    ROUND(AVG(login_velocity_24h), 2) AS avg_login_velocity,
    ROUND(AVG(device_trust_score), 4) AS avg_device_trust_score,
    ROUND(AVG(ip_risk_score), 4) AS avg_ip_risk_score,
    ROUND(AVG(merchant_risk_score), 4) AS avg_merchant_risk_score
FROM transactions
GROUP BY batch_file_name, batch_type
ORDER BY batch_file_name;


-- Missing value check by batch
SELECT
    batch_file_name,
    SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS missing_amount,
    SUM(CASE WHEN login_velocity_24h IS NULL THEN 1 ELSE 0 END) AS missing_login_velocity,
    SUM(CASE WHEN device_trust_score IS NULL THEN 1 ELSE 0 END) AS missing_device_trust_score,
    SUM(CASE WHEN ip_risk_score IS NULL THEN 1 ELSE 0 END) AS missing_ip_risk_score,
    SUM(CASE WHEN merchant_risk_score IS NULL THEN 1 ELSE 0 END) AS missing_merchant_risk_score
FROM transactions
GROUP BY batch_file_name
ORDER BY batch_file_name;


-- High-risk transactions by batch
SELECT
    batch_file_name,
    COUNT(*) AS high_risk_transactions,
    ROUND(AVG(fraud_probability_simulated), 4) AS avg_simulated_risk
FROM transactions
WHERE fraud_probability_simulated >= 0.30
GROUP BY batch_file_name
ORDER BY high_risk_transactions DESC;