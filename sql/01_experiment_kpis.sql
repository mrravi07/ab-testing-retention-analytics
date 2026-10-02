
-- Experiment KPI summary by variant
-- Database: ab_testing_db
-- Schema: analytics

WITH user_orders AS (
    SELECT
        user_id,
        COUNT(*) AS order_count,
        SUM(revenue) AS total_revenue
    FROM analytics.orders
    GROUP BY user_id
),
user_level AS (
    SELECT
        u.user_id,
        a.variant,
        COALESCE(o.order_count, 0) AS order_count,
        COALESCE(o.total_revenue, 0) AS total_revenue
    FROM analytics.users u
    INNER JOIN analytics.experiment_assignments a
        ON u.user_id = a.user_id
    LEFT JOIN user_orders o
        ON u.user_id = o.user_id
)
SELECT
    variant,
    COUNT(*) AS users,
    COUNT(*) FILTER (
        WHERE order_count > 0
    ) AS converters,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE order_count > 0)
        / NULLIF(COUNT(*), 0),
        3
    ) AS conversion_rate_pct,
    SUM(total_revenue)::NUMERIC(12, 2) AS total_revenue,
    ROUND(
        SUM(total_revenue) / NULLIF(COUNT(*), 0),
        4
    ) AS revenue_per_user,
    ROUND(
        SUM(total_revenue)
        / NULLIF(SUM(order_count), 0),
        2
    ) AS average_order_value
FROM user_level
GROUP BY variant
ORDER BY variant;

