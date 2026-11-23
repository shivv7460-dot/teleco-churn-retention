-- 1. Overall churn summary
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn IN (1, 'True', 'true', 'yes', 'Yes') THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN churn IN (1, 'True', 'true', 'yes', 'Yes') THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- 2. Churn by international plan
SELECT
    international_plan,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(CASE WHEN churn IN (1, 'True', 'true', 'yes', 'Yes') THEN 1.0 ELSE 0.0 END), 2) AS churn_rate_pct
FROM customers
GROUP BY international_plan
ORDER BY churn_rate_pct DESC;

-- 3. Churn by customer service calls
SELECT
    customer_service_calls,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(CASE WHEN churn IN (1, 'True', 'true', 'yes', 'Yes') THEN 1.0 ELSE 0.0 END), 2) AS churn_rate_pct
FROM customers
GROUP BY customer_service_calls
ORDER BY customer_service_calls;