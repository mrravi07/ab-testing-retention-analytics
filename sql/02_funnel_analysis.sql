
-- Funnel analysis by experiment variant
-- Each stage counts distinct users, not event occurrences.

WITH funnel_stages AS (
    SELECT
        a.variant,
        COUNT(DISTINCT e.user_id) FILTER (
            WHERE e.event_name = 'product_view'
        ) AS product_view_users,
        COUNT(DISTINCT e.user_id) FILTER (
            WHERE e.event_name = 'add_to_cart'
        ) AS add_to_cart_users,
        COUNT(DISTINCT e.user_id) FILTER (
            WHERE e.event_name = 'checkout_started'
        ) AS checkout_users,
        COUNT(DISTINCT e.user_id) FILTER (
            WHERE e.event_name = 'purchase'
        ) AS purchase_users
    FROM analytics.experiment_assignments a
    LEFT JOIN analytics.events e
        ON a.user_id = e.user_id
    GROUP BY a.variant
)
SELECT
    variant,
    product_view_users,
    add_to_cart_users,
    checkout_users,
    purchase_users,

    ROUND(
        100.0 * add_to_cart_users
        / NULLIF(product_view_users, 0),
        2
    ) AS view_to_cart_pct,

    ROUND(
        100.0 * checkout_users
        / NULLIF(add_to_cart_users, 0),
        2
    ) AS cart_to_checkout_pct,

    ROUND(
        100.0 * purchase_users
        / NULLIF(checkout_users, 0),
        2
    ) AS checkout_to_purchase_pct,

    ROUND(
        100.0 * purchase_users
        / NULLIF(product_view_users, 0),
        2
    ) AS overall_view_to_purchase_pct

FROM funnel_stages
ORDER BY variant;

