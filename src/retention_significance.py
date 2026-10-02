
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "reports" / "cohort" / "weekly_cohort_retention.csv"
OUTPUT_DIR = ROOT_DIR / "reports" / "cohort"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Week 0 is excluded because it is guaranteed by the signup event.
TEST_WEEKS = [1, 2, 4, 8, 12]
ALPHA = 0.05


def main():
    print("=" * 70)
    print("RETENTION STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE, parse_dates=["signup_week"])

    results = []

    for week in TEST_WEEKS:
        week_data = df[
            (df["cohort_age_week"] == week)
            & df["retained_users"].notna()
            & df["retention_rate"].notna()
        ].copy()

        control = week_data[week_data["variant"] == "Control"]
        treatment = week_data[week_data["variant"] == "Treatment"]

        control_users = int(control["cohort_users"].sum())
        treatment_users = int(treatment["cohort_users"].sum())

        control_retained = int(control["retained_users"].sum())
        treatment_retained = int(treatment["retained_users"].sum())

        if control_users == 0 or treatment_users == 0:
            print(f"Skipping Week {week}: no eligible users.")
            continue

        control_rate = control_retained / control_users
        treatment_rate = treatment_retained / treatment_users

        # Two-sided two-proportion z-test.
        z_stat, p_value = proportions_ztest(
            count=[treatment_retained, control_retained],
            nobs=[treatment_users, control_users],
            alternative="two-sided",
        )

        difference = treatment_rate - control_rate

        # Approximate unpooled 95% confidence interval.
        standard_error = np.sqrt(
            treatment_rate * (1 - treatment_rate) / treatment_users
            + control_rate * (1 - control_rate) / control_users
        )

        ci_lower = difference - 1.96 * standard_error
        ci_upper = difference + 1.96 * standard_error

        results.append(
            {
                "week": week,
                "control_users": control_users,
                "control_retained": control_retained,
                "control_retention_pct": control_rate * 100,
                "treatment_users": treatment_users,
                "treatment_retained": treatment_retained,
                "treatment_retention_pct": treatment_rate * 100,
                "absolute_lift_pp": difference * 100,
                "ci_lower_pp": ci_lower * 100,
                "ci_upper_pp": ci_upper * 100,
                "z_statistic": z_stat,
                "p_value": p_value,
            }
        )

    if not results:
        raise ValueError("No eligible retention comparisons were found.")

    results_df = pd.DataFrame(results)

    # Correct for multiple hypothesis tests.
    reject, adjusted_pvalues, _, _ = multipletests(
        results_df["p_value"],
        alpha=ALPHA,
        method="holm",
    )

    results_df["holm_adjusted_p_value"] = adjusted_pvalues
    results_df["significant_after_holm"] = reject

    results_df["interpretation"] = np.where(
        results_df["significant_after_holm"]
        & (results_df["absolute_lift_pp"] > 0),
        "Higher Treatment retention",
        np.where(
            results_df["significant_after_holm"]
            & (results_df["absolute_lift_pp"] < 0),
            "Lower Treatment retention",
            "No statistically significant difference",
        ),
    )

    output_file = OUTPUT_DIR / "retention_significance_results.csv"
    results_df.to_csv(output_file, index=False)

    display_columns = [
        "week",
        "control_retention_pct",
        "treatment_retention_pct",
        "absolute_lift_pp",
        "ci_lower_pp",
        "ci_upper_pp",
        "p_value",
        "holm_adjusted_p_value",
        "significant_after_holm",
        "interpretation",
    ]

    print("\nRETENTION COMPARISON")
    print(
        results_df[display_columns]
        .round(5)
        .to_string(index=False)
    )

    print("\nINTERPRETATION")
    print(
        "A Holm-adjusted p-value below 0.05 indicates a statistically "
        "significant difference for that retention period."
    )
    print(
        "A non-significant result does not prove that the variants "
        "have identical retention."
    )

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
