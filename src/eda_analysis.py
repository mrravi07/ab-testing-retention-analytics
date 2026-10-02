
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/eda")
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    users = pd.read_csv(
        DATA_DIR / "users.csv",
        parse_dates=["signup_date"],
    )

    assignments = pd.read_csv(
        DATA_DIR / "experiment_assignments.csv",
        parse_dates=["assigned_at"],
    )

    events = pd.read_csv(
        DATA_DIR / "events.csv",
        parse_dates=["event_timestamp"],
    )

    orders = pd.read_csv(
        DATA_DIR / "orders.csv",
        parse_dates=["order_date"],
    )

    return users, assignments, events, orders


# ============================================================
# 2. CREATE USER-LEVEL ANALYTICS TABLE
# ============================================================

def build_user_metrics(users, assignments, events, orders):
    """Create one row per randomized user."""

    user_metrics = (
        users.merge(
            assignments[["user_id", "variant"]],
            on="user_id",
            how="left",
            validate="one_to_one",
        )
    )

    # User-level funnel indicators
    event_flags = (
        events.assign(
            product_view=events["event_name"].eq("product_view"),
            add_to_cart=events["event_name"].eq("add_to_cart"),
            checkout_started=events["event_name"].eq(
                "checkout_started"
            ),
            purchase_event=events["event_name"].eq("purchase"),
        )
        .groupby("user_id")[
            [
                "product_view",
                "add_to_cart",
                "checkout_started",
                "purchase_event",
            ]
        ]
        .any()
        .reset_index()
    )

    user_metrics = user_metrics.merge(
        event_flags,
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    flag_columns = [
        "product_view",
        "add_to_cart",
        "checkout_started",
        "purchase_event",
    ]

    user_metrics[flag_columns] = (
        user_metrics[flag_columns].fillna(False).astype(bool)
    )

    # Order-level metrics aggregated to the user
    order_metrics = (
        orders.groupby("user_id")
        .agg(
            order_count=("order_id", "count"),
            total_revenue=("revenue", "sum"),
            average_order_value=("revenue", "mean"),
        )
        .reset_index()
    )

    user_metrics = user_metrics.merge(
        order_metrics,
        on="user_id",
        how="left",
        validate="one_to_one",
    )

    user_metrics["order_count"] = (
        user_metrics["order_count"].fillna(0).astype(int)
    )

    user_metrics["total_revenue"] = (
        user_metrics["total_revenue"].fillna(0.0)
    )

    user_metrics["average_order_value"] = (
        user_metrics["average_order_value"].fillna(0.0)
    )

    user_metrics["converted"] = (
        user_metrics["order_count"] > 0
    )

    # Cohort attributes
    user_metrics["signup_week"] = (
        user_metrics["signup_date"]
        .dt.to_period("W-SUN")
        .astype(str)
    )

    user_metrics["signup_month"] = (
        user_metrics["signup_date"]
        .dt.to_period("M")
        .astype(str)
    )

    return user_metrics


# ============================================================
# 3. EXPERIMENT KPI SUMMARY
# ============================================================

def experiment_summary(user_metrics):
    summary = (
        user_metrics.groupby("variant")
        .agg(
            users=("user_id", "nunique"),
            converters=("converted", "sum"),
            total_revenue=("total_revenue", "sum"),
            total_orders=("order_count", "sum"),
        )
        .reset_index()
    )

    summary["conversion_rate"] = (
        summary["converters"] / summary["users"]
    )

    summary["revenue_per_user"] = (
        summary["total_revenue"] / summary["users"]
    )

    summary["average_order_value"] = (
        summary["total_revenue"]
        / summary["total_orders"].replace(0, np.nan)
    )

    summary["conversion_rate_pct"] = (
        summary["conversion_rate"] * 100
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT KPI SUMMARY")
    print("=" * 65)

    print(
        summary[
            [
                "variant",
                "users",
                "converters",
                "conversion_rate_pct",
                "total_revenue",
                "revenue_per_user",
                "average_order_value",
            ]
        ].round(3).to_string(index=False)
    )

    summary.to_csv(
        OUTPUT_DIR / "experiment_kpi_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# 4. FUNNEL SUMMARY
# ============================================================

def funnel_summary(user_metrics):
    funnel_columns = {
        "Product View": "product_view",
        "Add to Cart": "add_to_cart",
        "Checkout Started": "checkout_started",
        "Purchase": "purchase_event",
    }

    rows = []

    for variant, group in user_metrics.groupby("variant"):
        previous_count = None

        for stage, column in funnel_columns.items():
            users_at_stage = int(group[column].sum())

            stage_conversion = (
                users_at_stage / len(group)
                if len(group) else 0
            )

            step_conversion = (
                users_at_stage / previous_count
                if previous_count else np.nan
            )

            rows.append({
                "variant": variant,
                "stage": stage,
                "users": users_at_stage,
                "conversion_from_assigned_users": stage_conversion,
                "conversion_from_previous_stage": step_conversion,
            })

            previous_count = users_at_stage

    funnel = pd.DataFrame(rows)

    funnel.to_csv(
        OUTPUT_DIR / "funnel_summary.csv",
        index=False,
    )

    print("\nFUNNEL SUMMARY")
    print(funnel.round(4).to_string(index=False))

    return funnel


# ============================================================
# 5. VISUALIZATIONS
# ============================================================

def plot_conversion(summary):
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=summary,
        x="variant",
        y="conversion_rate_pct",
        hue="variant",
        legend=False,
    )

    plt.title("User-Level Conversion Rate by Experiment Group")
    plt.xlabel("Experiment Group")
    plt.ylabel("Conversion Rate (%)")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "conversion_rate.png",
        dpi=150,
    )
    plt.close()


def plot_revenue(summary):
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=summary,
        x="variant",
        y="revenue_per_user",
        hue="variant",
        legend=False,
    )

    plt.title("Revenue per Assigned User")
    plt.xlabel("Experiment Group")
    plt.ylabel("Revenue per User")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "revenue_per_user.png",
        dpi=150,
    )
    plt.close()


def plot_funnel(funnel):
    pivot = funnel.pivot(
        index="stage",
        columns="variant",
        values="conversion_from_assigned_users",
    )

    pivot = pivot * 100

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("User Funnel by Experiment Group")
    ax.set_xlabel("Funnel Stage")
    ax.set_ylabel("Percentage of Assigned Users")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "funnel_comparison.png",
        dpi=150,
    )
    plt.close()


# ============================================================
# 6. SEGMENT ANALYSIS
# ============================================================

def segment_analysis(user_metrics):
    rows = []

    for dimension in [
        "device",
        "country",
        "acquisition_channel",
    ]:
        grouped = (
            user_metrics.groupby(
                [dimension, "variant"],
                observed=True,
            )
            .agg(
                users=("user_id", "nunique"),
                converters=("converted", "sum"),
                revenue=("total_revenue", "sum"),
            )
            .reset_index()
        )

        grouped["conversion_rate"] = (
            grouped["converters"] / grouped["users"]
        )

        grouped["revenue_per_user"] = (
            grouped["revenue"] / grouped["users"]
        )

        grouped.insert(0, "segment_dimension", dimension)
        grouped = grouped.rename(
            columns={dimension: "segment"}
        )

        rows.append(grouped)

    result = pd.concat(rows, ignore_index=True)

    result.to_csv(
        OUTPUT_DIR / "segment_analysis.csv",
        index=False,
    )

    print("\nSEGMENT ANALYSIS")
    print(result.round(4).to_string(index=False))

    return result


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("Loading datasets...")

    users, assignments, events, orders = load_data()

    print("Building user-level metrics...")

    user_metrics = build_user_metrics(
        users,
        assignments,
        events,
        orders,
    )

    user_metrics.to_csv(
        OUTPUT_DIR / "user_metrics.csv",
        index=False,
    )

    summary = experiment_summary(user_metrics)
    funnel = funnel_summary(user_metrics)
    segment_analysis(user_metrics)

    plot_conversion(summary)
    plot_revenue(summary)
    plot_funnel(funnel)

    print("\nEDA completed successfully.")
    print(f"Tables saved in: {OUTPUT_DIR.resolve()}")
    print(f"Charts saved in: {CHART_DIR.resolve()}")


if __name__ == "__main__":
    main()
