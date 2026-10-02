
from pathlib import Path

import pandas as pd
from scipy.stats import chisquare


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw")

EXPECTED_VARIANTS = {"Control", "Treatment"}

REQUIRED_COLUMNS = {
    "users.csv": {
        "user_id",
        "signup_date",
        "country",
        "device",
        "acquisition_channel",
    },
    "experiment_assignments.csv": {
        "user_id",
        "experiment_id",
        "variant",
        "assigned_at",
    },
    "events.csv": {
        "event_id",
        "user_id",
        "session_id",
        "event_name",
        "event_timestamp",
    },
    "orders.csv": {
        "order_id",
        "user_id",
        "order_date",
        "revenue",
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_datasets():
    """Load all raw CSV datasets."""

    datasets = {}

    for filename, required_columns in REQUIRED_COLUMNS.items():
        file_path = DATA_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{filename} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        datasets[filename] = df

        print(f"Loaded {filename}: {len(df):,} rows")

    return datasets


# ============================================================
# 1. NULL AND DUPLICATE CHECKS
# ============================================================

def check_nulls_and_duplicates(datasets):
    """Check missing values and duplicate primary keys."""

    primary_keys = {
        "users.csv": "user_id",
        "experiment_assignments.csv": "user_id",
        "events.csv": "event_id",
        "orders.csv": "order_id",
    }

    errors = []

    print("\n--- Null and Duplicate Checks ---")

    for filename, df in datasets.items():
        null_count = int(df.isnull().sum().sum())

        print(f"{filename}: total missing cells = {null_count:,}")

        if null_count > 0:
            errors.append(
                f"{filename} contains {null_count} missing cells"
            )

        key = primary_keys[filename]
        duplicate_count = int(df[key].duplicated().sum())

        print(
            f"{filename}: duplicate {key} = "
            f"{duplicate_count:,}"
        )

        if duplicate_count > 0:
            errors.append(
                f"{filename} contains duplicate {key} values"
            )

    return errors


# ============================================================
# 2. REFERENTIAL INTEGRITY
# ============================================================

def check_referential_integrity(datasets):
    """Check that related records reference existing users."""

    users = datasets["users.csv"]
    assignments = datasets["experiment_assignments.csv"]
    events = datasets["events.csv"]
    orders = datasets["orders.csv"]

    errors = []

    user_ids = set(users["user_id"])
    assigned_user_ids = set(assignments["user_id"])

    checks = {
        "Assignments referencing unknown users":
            assigned_user_ids - user_ids,
        "Events referencing unknown users":
            set(events["user_id"]) - user_ids,
        "Orders referencing unknown users":
            set(orders["user_id"]) - user_ids,
    }

    print("\n--- Referential Integrity ---")

    for check_name, invalid_ids in checks.items():
        print(f"{check_name}: {len(invalid_ids):,}")

        if invalid_ids:
            errors.append(check_name)

    missing_assignments = user_ids - assigned_user_ids

    print(f"Users without assignments: {len(missing_assignments):,}")

    if missing_assignments:
        errors.append("Some users have no experiment assignment")

    return errors


# ============================================================
# 3. EXPERIMENT ASSIGNMENT AND SRM CHECK
# ============================================================

def check_experiment_assignments(datasets):
    """
    Check valid variants, one assignment per user,
    and sample-ratio mismatch (SRM).
    """

    assignments = datasets["experiment_assignments.csv"]
    errors = []

    print("\n--- Experiment Assignment Checks ---")

    actual_variants = set(assignments["variant"].unique())

    if actual_variants != EXPECTED_VARIANTS:
        errors.append(
            f"Unexpected experiment variants: {actual_variants}"
        )

    counts = (
        assignments["variant"]
        .value_counts()
        .reindex(["Control", "Treatment"], fill_value=0)
    )

    print(counts.to_string())

    observed = counts.to_numpy()
    expected = [len(assignments) / 2, len(assignments) / 2]

    statistic, p_value = chisquare(
        f_obs=observed,
        f_exp=expected,
    )

    print(f"\nSRM chi-square statistic: {statistic:.4f}")
    print(f"SRM p-value: {p_value:.6f}")

    if p_value < 0.001:
        errors.append(
            "Potential sample-ratio mismatch detected"
        )
        print("SRM status: INVESTIGATE")
    else:
        print("SRM status: No significant mismatch detected")

    return errors


# ============================================================
# 4. DATE AND REVENUE CHECKS
# ============================================================

def check_dates_and_revenue(datasets):
    """Check timestamps, signup chronology, and revenue."""

    users = datasets["users.csv"].copy()
    assignments = datasets["experiment_assignments.csv"].copy()
    events = datasets["events.csv"].copy()
    orders = datasets["orders.csv"].copy()

    errors = []

    users["signup_date"] = pd.to_datetime(
        users["signup_date"], errors="coerce"
    )
    assignments["assigned_at"] = pd.to_datetime(
        assignments["assigned_at"], errors="coerce"
    )
    events["event_timestamp"] = pd.to_datetime(
        events["event_timestamp"], errors="coerce"
    )
    orders["order_date"] = pd.to_datetime(
        orders["order_date"], errors="coerce"
    )

    print("\n--- Date and Revenue Checks ---")

    date_columns = [
        (users, "signup_date", "users.csv"),
        (assignments, "assigned_at", "experiment_assignments.csv"),
        (events, "event_timestamp", "events.csv"),
        (orders, "order_date", "orders.csv"),
    ]

    for df, column, filename in date_columns:
        invalid_dates = int(df[column].isna().sum())

        print(f"{filename}: invalid {column} = {invalid_dates:,}")

        if invalid_dates:
            errors.append(f"{filename} has invalid timestamps")

    if users["signup_date"].isna().any():
        return errors

    if assignments["assigned_at"].isna().any():
        return errors

    if events["event_timestamp"].isna().any():
        return errors

    if orders["order_date"].isna().any():
        return errors

    user_dates = users.set_index("user_id")["signup_date"]

    event_signup_dates = events["user_id"].map(user_dates)
    order_signup_dates = orders["user_id"].map(user_dates)

    invalid_events = (
        events["event_timestamp"] < event_signup_dates
    )

    invalid_orders = orders["order_date"] < order_signup_dates

    print(f"Events before signup: {invalid_events.sum():,}")
    print(f"Orders before signup: {invalid_orders.sum():,}")

    if invalid_events.any():
        errors.append("Events exist before user signup")

    if invalid_orders.any():
        errors.append("Orders exist before user signup")

    if (orders["revenue"] <= 0).any():
        errors.append("Non-positive order revenue detected")

    if not pd.api.types.is_numeric_dtype(orders["revenue"]):
        errors.append("Revenue must be numeric")

    if (orders["order_date"] > pd.Timestamp("2026-06-30 23:59:59")).any():
        errors.append("Orders exist beyond the observation window")

    return errors


# ============================================================
# 5. EVENT AND ORDER CONSISTENCY
# ============================================================

def check_event_order_consistency(datasets):
    """Check event types and purchase/order consistency."""

    events = datasets["events.csv"].copy()
    orders = datasets["orders.csv"].copy()

    errors = []

    valid_events = {
        "product_view",
        "add_to_cart",
        "checkout_started",
        "purchase",
    }

    actual_events = set(events["event_name"].dropna().unique())
    invalid_events = actual_events - valid_events

    print("\n--- Event and Order Consistency ---")

    print(f"Unknown event types: {len(invalid_events):,}")

    if invalid_events:
        errors.append(f"Unknown event types: {invalid_events}")

    purchase_events = events[
        events["event_name"] == "purchase"
    ]

    print(f"Purchase events: {len(purchase_events):,}")
    print(f"Orders: {len(orders):,}")

    if len(purchase_events) != len(orders):
        errors.append(
            "Purchase event count does not match order count"
        )

    # Each order must have exactly one matching purchase event
    event_purchase_counts = purchase_events["user_id"].value_counts()
    order_counts = orders["user_id"].value_counts()

    if (event_purchase_counts > 1).any():
        errors.append(
            "Some users have multiple purchase events"
        )

    if (order_counts > 1).any():
        errors.append(
            "Some users have multiple orders"
        )

    purchase_users = set(purchase_events["user_id"])
    order_users = set(orders["user_id"])

    if purchase_users != order_users:
        errors.append(
            "Purchase-event users and order users do not match"
        )

    # Validate the event sequence within each session.
    event_order = {
        "product_view": 1,
        "add_to_cart": 2,
        "checkout_started": 3,
        "purchase": 4,
    }

    events["event_rank"] = events["event_name"].map(event_order)

    session_events = events.sort_values(
        ["session_id", "event_timestamp"]
    )

    for session_id, group in session_events.groupby("session_id"):
        ranks = group["event_rank"].tolist()

        if ranks != sorted(ranks):
            errors.append(
                f"Invalid funnel event sequence in session {session_id}"
            )
            break

    print("Event and order checks completed.")

    return errors


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("ADVANCED DATA QUALITY VALIDATION")
    print("=" * 60)

    datasets = load_datasets()

    errors = []

    checks = [
        check_nulls_and_duplicates,
        check_referential_integrity,
        check_experiment_assignments,
        check_dates_and_revenue,
        check_event_order_consistency,
    ]

    for check in checks:
        errors.extend(check(datasets))

    print("\n" + "=" * 60)
    print("FINAL VALIDATION REPORT")
    print("=" * 60)

    if errors:
        print(f"FAILED: {len(errors)} issue(s) found.")

        for index, error in enumerate(errors, start=1):
            print(f"{index}. {error}")

        raise SystemExit(1)

    print("PASSED: All data quality checks completed successfully.")
    print("Datasets are ready for exploratory analysis.")


if __name__ == "__main__":
    main()
