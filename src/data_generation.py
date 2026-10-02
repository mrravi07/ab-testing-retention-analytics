
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
N_USERS = 50_000

START_DATE = pd.Timestamp("2025-11-01")
END_DATE = pd.Timestamp("2026-06-30")

OUTPUT_DIR = Path("data/raw")

rng = np.random.default_rng(SEED)


# ============================================================
# 1. GENERATE USERS
# ============================================================

def generate_users(n_users: int = N_USERS) -> pd.DataFrame:
    """Generate synthetic user profiles and signup dates."""

    signup_days = rng.integers(
        0,
        (END_DATE - START_DATE).days + 1,
        size=n_users,
    )

    users = pd.DataFrame({
        "user_id": [
            f"U{i:06d}" for i in range(1, n_users + 1)
        ],
        "signup_date": (
            START_DATE + pd.to_timedelta(signup_days, unit="D")
        ),
        "country": rng.choice(
            ["India", "USA", "UK", "Canada", "Australia"],
            size=n_users,
            p=[0.50, 0.20, 0.12, 0.10, 0.08],
        ),
        "device": rng.choice(
            ["Mobile", "Desktop", "Tablet"],
            size=n_users,
            p=[0.65, 0.28, 0.07],
        ),
        "acquisition_channel": rng.choice(
            ["Organic", "Paid Search", "Social", "Email", "Referral"],
            size=n_users,
            p=[0.30, 0.25, 0.20, 0.15, 0.10],
        ),
    })

    users["signup_date"] = pd.to_datetime(users["signup_date"])

    return users.sort_values(
        "signup_date"
    ).reset_index(drop=True)


# ============================================================
# 2. GENERATE RANDOMIZED EXPERIMENT ASSIGNMENTS
# ============================================================

def generate_assignments(
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Randomly assign users to balanced control/treatment groups."""

    n = len(users)

    variants = np.array(
        ["Control"] * (n // 2)
        + ["Treatment"] * (n - n // 2)
    )

    rng.shuffle(variants)

    assignments = pd.DataFrame({
        "user_id": users["user_id"].to_numpy(),
        "experiment_id": "checkout_redesign_v1",
        "variant": variants,
        "assigned_at": users["signup_date"].to_numpy(),
    })

    assignments["assigned_at"] = pd.to_datetime(
        assignments["assigned_at"]
    )

    return assignments


# ============================================================
# 3. GENERATE EVENTS AND ORDERS
# ============================================================

def generate_events(
    users: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate product views, cart, checkout and purchase events.

    Synthetic experiment assumptions:
    - Control purchase probability per eligible checkout: 12%
    - Treatment purchase probability per eligible checkout: 14%
    - At most one order per user
    """

    assignment_map = assignments.set_index(
        "user_id"
    )["variant"]

    event_rows = []
    order_rows = []

    purchased_users = set()

    event_counter = 1
    order_counter = 1

    for user in users.itertuples(index=False):
        user_id = user.user_id
        signup = user.signup_date
        variant = assignment_map[user_id]

        # ----------------------------------------------------
        # Generate the initial session and repeat sessions
        # ----------------------------------------------------

        session_dates = [signup]

        available_days = max(
            0,
            (END_DATE - signup).days,
        )

        if available_days > 0:
            repeat_sessions = rng.poisson(
                lam=2.5 * min(available_days / 120, 1.0)
            )

            for _ in range(repeat_sessions):
                day_offset = int(
                    rng.integers(1, available_days + 1)
                )

                session_dates.append(
                    signup + pd.Timedelta(days=day_offset)
                )

        session_dates.sort()

        # ----------------------------------------------------
        # Generate funnel events for each session
        # ----------------------------------------------------

        for session_date in session_dates:
            session_id = f"S{event_counter:09d}"

            # Product view
            event_rows.append({
                "event_id": f"E{event_counter:09d}",
                "user_id": user_id,
                "session_id": session_id,
                "event_name": "product_view",
                "event_timestamp": (
                    session_date + pd.Timedelta(minutes=1)
                ),
            })
            event_counter += 1

            # Add to cart: 35% probability
            if rng.random() >= 0.35:
                continue

            event_rows.append({
                "event_id": f"E{event_counter:09d}",
                "user_id": user_id,
                "session_id": session_id,
                "event_name": "add_to_cart",
                "event_timestamp": (
                    session_date + pd.Timedelta(minutes=5)
                ),
            })
            event_counter += 1

            # Checkout started: 60% probability after cart
            if rng.random() >= 0.60:
                continue

            event_rows.append({
                "event_id": f"E{event_counter:09d}",
                "user_id": user_id,
                "session_id": session_id,
                "event_name": "checkout_started",
                "event_timestamp": (
                    session_date + pd.Timedelta(minutes=10)
                ),
            })
            event_counter += 1

            # At most one order per user
            if user_id in purchased_users:
                continue

            # Simulated treatment effect
            purchase_probability = (
                0.12 if variant == "Control" else 0.14
            )

            if rng.random() >= purchase_probability:
                continue

            # Generate positive order revenue
            revenue = round(
                float(
                    rng.lognormal(
                        mean=np.log(55),
                        sigma=0.55,
                    )
                ),
                2,
            )

            purchase_time = (
                session_date + pd.Timedelta(minutes=15)
            )

            # Purchase event
            event_rows.append({
                "event_id": f"E{event_counter:09d}",
                "user_id": user_id,
                "session_id": session_id,
                "event_name": "purchase",
                "event_timestamp": purchase_time,
            })
            event_counter += 1

            # Corresponding order record
            order_rows.append({
                "order_id": f"O{order_counter:07d}",
                "user_id": user_id,
                "order_date": purchase_time,
                "revenue": revenue,
            })
            order_counter += 1

            purchased_users.add(user_id)

    # --------------------------------------------------------
    # Convert records into DataFrames
    # --------------------------------------------------------

    events = pd.DataFrame(
        event_rows,
        columns=[
            "event_id",
            "user_id",
            "session_id",
            "event_name",
            "event_timestamp",
        ],
    )

    orders = pd.DataFrame(
        order_rows,
        columns=[
            "order_id",
            "user_id",
            "order_date",
            "revenue",
        ],
    )

    events["event_timestamp"] = pd.to_datetime(
        events["event_timestamp"]
    )
    orders["order_date"] = pd.to_datetime(
        orders["order_date"]
    )

    events = events.sort_values(
        ["event_timestamp", "event_id"]
    ).reset_index(drop=True)

    orders = orders.sort_values(
        ["order_date", "order_id"]
    ).reset_index(drop=True)

    return events, orders


# ============================================================
# 4. VALIDATE DATASETS
# ============================================================

def validate_datasets(
    users: pd.DataFrame,
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    orders: pd.DataFrame,
) -> None:
    """Run basic integrity and consistency checks."""

    # Unique primary keys
    assert users["user_id"].is_unique, "Duplicate user IDs"
    assert assignments["user_id"].is_unique, (
        "Duplicate experiment assignments"
    )
    assert events["event_id"].is_unique, "Duplicate event IDs"
    assert orders["order_id"].is_unique, "Duplicate order IDs"

    # Referential integrity
    user_ids = set(users["user_id"])

    assert set(assignments["user_id"]) == user_ids
    assert set(events["user_id"]).issubset(user_ids)
    assert set(orders["user_id"]).issubset(user_ids)

    # Assignment validation
    assert set(assignments["variant"].unique()) == {
        "Control",
        "Treatment",
    }

    group_counts = assignments["variant"].value_counts()

    assert group_counts["Control"] + group_counts["Treatment"] == (
        len(users)
    )

    # Event validation
    valid_events = {
        "product_view",
        "add_to_cart",
        "checkout_started",
        "purchase",
    }

    assert set(events["event_name"].unique()).issubset(
        valid_events
    )

    # Revenue validation
    assert (orders["revenue"] > 0).all(), (
        "Non-positive revenue detected"
    )

    # Date validation
    assert users["signup_date"].between(
        START_DATE, END_DATE
    ).all()

    assert (
        events["event_timestamp"].dt.normalize() <= END_DATE
    ).all()

    assert (
        orders["order_date"].dt.normalize() <= END_DATE
    ).all()

    # Every order must have a corresponding purchase event
    purchase_users = set(
        events.loc[
            events["event_name"] == "purchase",
            "user_id",
        ]
    )

    assert set(orders["user_id"]).issubset(purchase_users)

    print("All basic data quality checks passed.")


# ============================================================
# 5. SAVE CSV FILES
# ============================================================

def save_datasets(
    users: pd.DataFrame,
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    orders: pd.DataFrame,
) -> None:
    """Save generated datasets into data/raw/."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "users.csv": users,
        "experiment_assignments.csv": assignments,
        "events.csv": events,
        "orders.csv": orders,
    }

    for filename, dataframe in datasets.items():
        file_path = OUTPUT_DIR / filename

        dataframe.to_csv(
            file_path,
            index=False,
        )

        print(
            f"Saved {file_path}: "
            f"{len(dataframe):,} rows"
        )


# ============================================================
# 6. PRINT DATASET SUMMARY
# ============================================================

def print_summary(
    users: pd.DataFrame,
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    orders: pd.DataFrame,
) -> None:
    """Print useful summary statistics."""

    print("\n" + "=" * 60)
    print("SYNTHETIC PRODUCT ANALYTICS DATA SUMMARY")
    print("=" * 60)

    print(f"Total users: {len(users):,}")
    print(f"Total events: {len(events):,}")
    print(f"Total orders: {len(orders):,}")

    print("\nExperiment group distribution:")
    print(
        assignments["variant"]
        .value_counts()
        .to_string()
    )

    print("\nEvent distribution:")
    print(
        events["event_name"]
        .value_counts()
        .to_string()
    )

    print("\nSignup date range:")
    print(
        f"{users['signup_date'].min().date()} "
        f"to {users['signup_date'].max().date()}"
    )

    print("\nRevenue summary:")
    if not orders.empty:
        print(f"Total revenue: ${orders['revenue'].sum():,.2f}")
        print(f"Average order value: ${orders['revenue'].mean():,.2f}")
    else:
        print("No orders generated.")

    print("\nOutput directory:", OUTPUT_DIR.resolve())
    print("=" * 60)


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main() -> None:
    print("Generating synthetic datasets...\n")

    users = generate_users()
    assignments = generate_assignments(users)

    events, orders = generate_events(
        users,
        assignments,
    )

    validate_datasets(
        users,
        assignments,
        events,
        orders,
    )

    save_datasets(
        users,
        assignments,
        events,
        orders,
    )

    print_summary(
        users,
        assignments,
        events,
        orders,
    )

    print("\nDataset generation completed successfully!")


if __name__ == "__main__":
    main()
