
-- Segment analysis by device and acquisition channel.
-- One row per segment and experiment variant.

WITH user_orders AS (
    SELECT
        user_id,
        COUNT(*) AS order_count,
        SUM(revenue) AS total_revenue
    FROM analytics.orders
    GROUP BY user_id
),
user_metrics AS (
    SELECT
        u.user_id,
        a.variant,
        u.device,
        u.acquisition_channel,
        COALESCE(o.order_count, 0) AS order_count,
        COALESCE(o.total_revenue, 0) AS total_revenue
    FROM analytics.users u
    JOIN analytics.experiment_assignments a
        ON u.user_id = a.user_id
    LEFT JOIN user_orders o
        ON u.user_id = o.user_id
),
segment_rows AS (
    SELECT
        'device' AS segment_type,
        device AS segment,
        variant,
        user_id,
        order_count,
        total_revenue
    FROM user_metrics

    UNION ALL

    SELECT
        'acquisition_channel' AS segment_type,
        acquisition_channel AS segment,
        variant,
        user_id,
        order_count,
        total_revenue
    FROM user_metrics
)
SELECT
    segment_type,
    segment,
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
    ROUND(
        SUM(total_revenue) / NULLIF(COUNT(*), 0),
        4
    ) AS revenue_per_user,
    ROUND(
        SUM(total_revenue)
        / NULLIF(SUM(order_count), 0),
        2
    ) AS average_order_value
FROM segment_rows
GROUP BY segment_type, segment, variant
ORDER BY segment_type, segment, variant;

