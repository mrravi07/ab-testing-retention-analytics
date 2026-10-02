
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports" / "cohort"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = RAW_DIR / "users.csv"
ASSIGNMENTS_FILE = RAW_DIR / "experiment_assignments.csv"
EVENTS_FILE = RAW_DIR / "events.csv"

# Week 0 is the signup week.
# Week 1 is the following calendar week relative to signup week.
MAX_COHORT_AGE_WEEKS = 16


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    users = pd.read_csv(USERS_FILE, parse_dates=["signup_date"])
    assignments = pd.read_csv(ASSIGNMENTS_FILE)
    events = pd.read_csv(EVENTS_FILE, parse_dates=["event_timestamp"])

    users["user_id"] = users["user_id"].astype(str)
    assignments["user_id"] = assignments["user_id"].astype(str)
    events["user_id"] = events["user_id"].astype(str)

    users = users.merge(
        assignments[["user_id", "variant"]],
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    if users["variant"].isna().any():
        raise ValueError("Some users have no experiment assignment.")

    if users["user_id"].duplicated().any():
        raise ValueError("Duplicate user IDs found in users data.")

    if events["event_timestamp"].isna().any():
        raise ValueError("Invalid event timestamps found.")

    return users, events


# ============================================================
# BUILD WEEKLY COHORTS
# ============================================================

def build_cohort_data(users, events):
    # Normalize timestamps to calendar dates.
    users["signup_date"] = users["signup_date"].dt.normalize()
    events["event_date"] = events["event_timestamp"].dt.normalize()

    # Assign each user to the Monday of their signup week.
    users["signup_week"] = (
        users["signup_date"]
        - pd.to_timedelta(users["signup_date"].dt.weekday, unit="D")
    )

    # Calculate the week in which each event occurred.
    events["event_week"] = (
        events["event_date"]
        - pd.to_timedelta(events["event_date"].dt.weekday, unit="D")
    )

    activity = events.merge(
        users[["user_id", "signup_date", "signup_week", "variant"]],
        on="user_id",
        how="inner",
        validate="many_to_one",
    )

    # Do not count events occurring before signup.
    activity = activity[
        activity["event_date"] >= activity["signup_date"]
    ].copy()

    activity["cohort_age_week"] = (
        (activity["event_week"] - activity["signup_week"]).dt.days // 7
    ).astype(int)

    activity = activity[
        activity["cohort_age_week"].between(
            0, MAX_COHORT_AGE_WEEKS
        )
    ].copy()

    # Count a user only once per cohort-age week, regardless of
    # how many events that user generated during the week.
    active_users = activity[
        ["user_id", "signup_week", "variant", "cohort_age_week"]
    ].drop_duplicates()

    return users, active_users


# ============================================================
# RETENTION MATRIX
# ============================================================

def calculate_retention(users, active_users):
    cohort_sizes = (
        users.groupby(["variant", "signup_week"])["user_id"]
        .nunique()
        .rename("cohort_users")
        .reset_index()
    )

    retained = (
        active_users.groupby(
            ["variant", "signup_week", "cohort_age_week"]
        )["user_id"]
        .nunique()
        .rename("retained_users")
        .reset_index()
    )

    retention = retained.merge(
        cohort_sizes,
        on=["variant", "signup_week"],
        how="left",
        validate="many_to_one",
    )

    retention["retention_rate"] = (
        retention["retained_users"] / retention["cohort_users"]
    )

    # Determine the latest complete calendar week in the event data.
    # Current partial week is excluded from mature-week comparisons.
    latest_event_date = active_users["signup_week"].max()

    # Use the maximum actual event week from source events instead
    # of assuming the current date is fully observed.
    latest_observed_week = active_users["signup_week"].max()

    # The cohort table is based on signup cohorts, while activity
    # is observed through the final date in the events dataset.
    # We calculate the actual final observed event week separately.
    return retention, cohort_sizes


def build_retention_matrix(users, active_users, events):
    cohort_sizes = (
        users.groupby(["variant", "signup_week"])["user_id"]
        .nunique()
        .rename("cohort_users")
        .reset_index()
    )

    retained = (
        active_users.groupby(
            ["variant", "signup_week", "cohort_age_week"]
        )["user_id"]
        .nunique()
        .rename("retained_users")
        .reset_index()
    )

    retention = retained.merge(
        cohort_sizes,
        on=["variant", "signup_week"],
        how="left",
        validate="many_to_one",
    )

    retention["retention_rate"] = (
        retention["retained_users"] / retention["cohort_users"]
    )

    # The dataset's actual observation end is the latest event date.
    last_event_date = events["event_date"].max()
    last_observed_week = (
        last_event_date
        - pd.to_timedelta(last_event_date.weekday(), unit="D")
    )

    cohort_weeks = sorted(users["signup_week"].unique())
    age_weeks = list(range(MAX_COHORT_AGE_WEEKS + 1))

    matrix_rows = []

    for _, cohort in cohort_sizes.iterrows():
        signup_week = cohort["signup_week"]
        variant = cohort["variant"]
        size = int(cohort["cohort_users"])

        for age in age_weeks:
            target_week = signup_week + pd.Timedelta(weeks=age)

            # A week is observable only if it has ended by the
            # last observed event date. This avoids labeling a
            # partial future period as zero retention.
            if target_week + pd.Timedelta(days=6) > last_event_date:
                rate = np.nan
                retained_count = np.nan
            else:
                match = retention[
                    (retention["variant"] == variant)
                    & (retention["signup_week"] == signup_week)
                    & (retention["cohort_age_week"] == age)
                ]

                if match.empty:
                    retained_count = 0
                    rate = 0.0
                else:
                    retained_count = int(match["retained_users"].iloc[0])
                    rate = retained_count / size

            matrix_rows.append(
                {
                    "variant": variant,
                    "signup_week": signup_week,
                    "cohort_age_week": age,
                    "cohort_users": size,
                    "retained_users": retained_count,
                    "retention_rate": rate,
                }
            )

    retention_matrix = pd.DataFrame(matrix_rows)

    return retention_matrix, cohort_sizes


# ============================================================
# EXPORT TABLES
# ============================================================

def save_outputs(retention_matrix, cohort_sizes):
    retention_matrix.to_csv(
        REPORT_DIR / "weekly_cohort_retention.csv",
        index=False,
    )

    cohort_sizes.to_csv(
        REPORT_DIR / "cohort_sizes.csv",
        index=False,
    )

    # Separate matrices for Control and Treatment.
    for variant in ["Control", "Treatment"]:
        subset = retention_matrix[
            retention_matrix["variant"] == variant
        ]

        matrix = subset.pivot(
            index="signup_week",
            columns="cohort_age_week",
            values="retention_rate",
        )

        matrix.to_csv(
            REPORT_DIR / f"{variant.lower()}_retention_matrix.csv"
        )

    print("\nCOHORT SIZES")
    print(cohort_sizes.to_string(index=False))

    print("\nRETENTION DATA SAMPLE")
    print(retention_matrix.head(20).to_string(index=False))


# ============================================================
# VISUALIZATION
# ============================================================

def plot_retention_heatmaps(retention_matrix):
    for variant in ["Control", "Treatment"]:
        subset = retention_matrix[
            retention_matrix["variant"] == variant
        ]

        matrix = subset.pivot(
            index="signup_week",
            columns="cohort_age_week",
            values="retention_rate",
        )

        matrix.index = pd.to_datetime(matrix.index).strftime("%Y-%m-%d")

        plt.figure(figsize=(15, 8))

        sns.heatmap(
            matrix * 100,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            linewidths=0.4,
            mask=matrix.isna(),
            cbar_kws={"label": "Weekly retention (%)"},
        )

        plt.title(f"{variant} — Weekly Cohort Retention")
        plt.xlabel("Weeks Since Signup")
        plt.ylabel("Signup Cohort Week")
        plt.tight_layout()

        output_file = REPORT_DIR / (
            f"{variant.lower()}_retention_heatmap.png"
        )

        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved heatmap: {output_file}")


# ============================================================
# COHORT RETENTION SUMMARY
# ============================================================

def create_retention_summary(retention_matrix):
    summary = (
        retention_matrix.groupby(
            ["variant", "cohort_age_week"]
        )
        .apply(
            lambda group: pd.Series(
                {
                    "eligible_cohorts": group["retention_rate"].notna().sum(),
                    "average_cohort_retention_pct": (
                        group["retention_rate"].mean() * 100
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    summary.to_csv(
        REPORT_DIR / "retention_by_cohort_age.csv",
        index=False,
    )

    print("\nRETENTION BY COHORT AGE")
    print(summary.to_string(index=False))

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("WEEKLY COHORT RETENTION ANALYSIS")
    print("=" * 65)

    users, events = load_data()

    print(f"Users loaded: {len(users):,}")
    print(f"Events loaded: {len(events):,}")

    users, active_users = build_cohort_data(users, events)

    retention_matrix, cohort_sizes = build_retention_matrix(
        users, active_users, events
    )

    save_outputs(retention_matrix, cohort_sizes)
    plot_retention_heatmaps(retention_matrix)
    create_retention_summary(retention_matrix)

    print("\nCOHORT ANALYSIS COMPLETED SUCCESSFULLY.")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()

