
-- Weekly cohort retention by experiment variant.
-- Week 0 = signup calendar week.
-- A user is retained if they have at least one event in that week.
-- Incomplete observation weeks are returned as NULL.

WITH cohort_sizes AS (
    SELECT
        a.variant,
        u.user_id,
        DATE_TRUNC('week', u.signup_date)::DATE AS signup_week
    FROM analytics.users u
    JOIN analytics.experiment_assignments a
        ON u.user_id = a.user_id
),
cohort_totals AS (
    SELECT
        variant,
        signup_week,
        COUNT(DISTINCT user_id) AS cohort_users
    FROM cohort_sizes
    GROUP BY variant, signup_week
),
active_users AS (
    SELECT DISTINCT
        c.variant,
        c.signup_week,
        e.user_id,
        (
            (
                DATE_TRUNC('week', e.event_timestamp)::DATE
                - c.signup_week
            ) / 7
        ) AS cohort_age_week
    FROM cohort_sizes c
    JOIN analytics.events e
        ON c.user_id = e.user_id
    WHERE e.event_timestamp::DATE >= (
        SELECT MIN(signup_date)
        FROM analytics.users
        WHERE user_id = c.user_id
    )
),
weekly_retained AS (
    SELECT
        variant,
        signup_week,
        cohort_age_week,
        COUNT(DISTINCT user_id) AS retained_users
    FROM active_users
    WHERE cohort_age_week BETWEEN 0 AND 16
    GROUP BY variant, signup_week, cohort_age_week
),
last_observation AS (
    SELECT MAX(event_timestamp::DATE) AS last_event_date
    FROM analytics.events
),
cohort_age_grid AS (
    SELECT
        ct.variant,
        ct.signup_week,
        ct.cohort_users,
        age.cohort_age_week,
        ct.signup_week + (age.cohort_age_week * 7) AS target_week
    FROM cohort_totals ct
    CROSS JOIN GENERATE_SERIES(0, 16) AS age(cohort_age_week)
)
SELECT
    g.variant,
    g.signup_week,
    g.cohort_age_week,
    g.cohort_users,

    CASE
        WHEN g.target_week + 6 <= o.last_event_date
        THEN COALESCE(r.retained_users, 0)
        ELSE NULL
    END AS retained_users,

    CASE
        WHEN g.target_week + 6 <= o.last_event_date
        THEN ROUND(
            100.0 * COALESCE(r.retained_users, 0)
            / NULLIF(g.cohort_users, 0),
            4
        )
        ELSE NULL
    END AS retention_rate_pct

FROM cohort_age_grid g
CROSS JOIN last_observation o
LEFT JOIN weekly_retained r
    ON g.variant = r.variant
    AND g.signup_week = r.signup_week
    AND g.cohort_age_week = r.cohort_age_week
ORDER BY
    g.variant,
    g.signup_week,
    g.cohort_age_week;

