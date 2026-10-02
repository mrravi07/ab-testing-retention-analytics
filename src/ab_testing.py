
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, norm
from statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_effectsize,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw")
EDA_DIR = Path("reports/eda")
OUTPUT_DIR = Path("reports/ab_testing")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_LEVEL = 0.95
ALPHA = 1 - CONFIDENCE_LEVEL

# Example business threshold:
# Require at least a 0.5 percentage-point absolute lift.
MIN_PRACTICAL_LIFT = 0.005

BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 42


# ============================================================
# 1. LOAD USER-LEVEL METRICS
# ============================================================

def load_data():
    """Load the user-level analysis table."""

    path = EDA_DIR / "user_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/eda_analysis.py first."
        )

    df = pd.read_csv(path)

    required = {
        "user_id",
        "variant",
        "converted",
        "total_revenue",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["user_id"].duplicated().any():
        raise ValueError(
            "Expected one row per randomized user."
        )

    if df["variant"].isna().any():
        raise ValueError("Missing experiment assignments.")

    df["converted"] = (
        df["converted"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["converted"].isna().any():
        raise ValueError("Invalid conversion values.")

    df["total_revenue"] = pd.to_numeric(
        df["total_revenue"],
        errors="raise",
    )

    if not np.isfinite(df["total_revenue"]).all():
        raise ValueError("Revenue contains non-finite values.")

    if (df["total_revenue"] < 0).any():
        raise ValueError("Negative revenue found.")

    if set(df["variant"].unique()) != {
        "Control", "Treatment"
    }:
        raise ValueError(
            "Expected Control and Treatment groups."
        )

    return df


# ============================================================
# 2. CALCULATE GROUP METRICS
# ============================================================

def calculate_group_metrics(df):
    """Calculate intention-to-treat metrics per assigned user."""

    rows = []

    for variant in ["Control", "Treatment"]:
        group = df.loc[df["variant"] == variant]

        n_users = len(group)
        converters = int(group["converted"].sum())

        conversion_rate = converters / n_users

        total_revenue = group["total_revenue"].sum()

        rows.append({
            "variant": variant,
            "users": n_users,
            "converters": converters,
            "non_converters": n_users - converters,
            "conversion_rate": conversion_rate,
            "conversion_rate_pct": conversion_rate * 100,
            "total_revenue": total_revenue,
            "revenue_per_user": total_revenue / n_users,
            "revenue_per_user_sd": (
                group["total_revenue"].std(ddof=1)
            ),
        })

    metrics = pd.DataFrame(rows).set_index("variant")

    print("\nGROUP METRICS")
    print(metrics.round(4).to_string())

    return metrics


# ============================================================
# 3. CONVERSION HYPOTHESIS TEST
# ============================================================

def conversion_test(metrics):
    """
    H0: Treatment conversion rate = Control conversion rate
    H1: Treatment conversion rate != Control conversion rate

    Two-sided pooled two-proportion z-test.
    """

    control = metrics.loc["Control"]
    treatment = metrics.loc["Treatment"]

    x_control = int(control["converters"])
    n_control = int(control["users"])

    x_treatment = int(treatment["converters"])
    n_treatment = int(treatment["users"])

    counts = np.array([x_treatment, x_control])
    sample_sizes = np.array([n_treatment, n_control])

    z_stat, p_value = proportions_ztest(
        count=counts,
        nobs=sample_sizes,
        alternative="two-sided",
    )

    p_control = x_control / n_control
    p_treatment = x_treatment / n_treatment

    absolute_lift = p_treatment - p_control

    relative_lift = (
        absolute_lift / p_control
        if p_control > 0
        else np.nan
    )

    # Cohen's h for two independent proportions
    cohens_h = proportion_effectsize(
        p_treatment,
        p_control,
    )

    # Unpooled Wald CI for the difference in proportions.
    # With these sample sizes, this is a useful approximation.
    standard_error = np.sqrt(
        p_control * (1 - p_control) / n_control
        + p_treatment * (1 - p_treatment) / n_treatment
    )

    critical_value = norm.ppf(1 - ALPHA / 2)

    ci_lower = absolute_lift - critical_value * standard_error
    ci_upper = absolute_lift + critical_value * standard_error

    statistically_significant = p_value < ALPHA

    practically_significant = (
        absolute_lift >= MIN_PRACTICAL_LIFT
    )

    results = {
        "control_conversion": p_control,
        "treatment_conversion": p_treatment,
        "absolute_lift": absolute_lift,
        "absolute_lift_pp": absolute_lift * 100,
        "relative_lift_pct": relative_lift * 100,
        "z_statistic": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_lower_pp": ci_lower * 100,
        "ci_upper_pp": ci_upper * 100,
        "cohens_h": cohens_h,
        "statistically_significant": statistically_significant,
        "practically_significant": practically_significant,
    }

    return results


# ============================================================
# 4. CHI-SQUARE TEST
# ============================================================

def chi_square_test(metrics):
    """Independent cross-check of the conversion comparison."""

    control = metrics.loc["Control"]
    treatment = metrics.loc["Treatment"]

    table = np.array([
        [
            int(control["converters"]),
            int(control["non_converters"]),
        ],
        [
            int(treatment["converters"]),
            int(treatment["non_converters"]),
        ],
    ])

    statistic, p_value, dof, expected = chi2_contingency(
        table,
        correction=False,
    )

    return {
        "chi_square_statistic": statistic,
        "chi_square_p_value": p_value,
        "degrees_of_freedom": dof,
        "minimum_expected_count": expected.min(),
    }


# ============================================================
# 5. REVENUE PER USER BOOTSTRAP
# ============================================================

def bootstrap_revenue_test(df):
    """
    Estimate the difference in revenue per assigned user.

    Resample users independently within each group.
    The percentile interval describes sampling uncertainty
    under the observed user-level revenue distributions.
    """

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    control = df.loc[
        df["variant"] == "Control", "total_revenue"
    ].to_numpy(dtype=float)

    treatment = df.loc[
        df["variant"] == "Treatment", "total_revenue"
    ].to_numpy(dtype=float)

    observed_difference = (
        treatment.mean() - control.mean()
    )

    differences = np.empty(BOOTSTRAP_ITERATIONS)

    for i in range(BOOTSTRAP_ITERATIONS):
        control_sample = rng.choice(
            control,
            size=len(control),
            replace=True,
        )

        treatment_sample = rng.choice(
            treatment,
            size=len(treatment),
            replace=True,
        )

        differences[i] = (
            treatment_sample.mean()
            - control_sample.mean()
        )

    ci_lower, ci_upper = np.quantile(
        differences,
        [ALPHA / 2, 1 - ALPHA / 2],
    )

    return {
        "control_revenue_per_user": control.mean(),
        "treatment_revenue_per_user": treatment.mean(),
        "revenue_per_user_difference": observed_difference,
        "revenue_ci_lower": ci_lower,
        "revenue_ci_upper": ci_upper,
        "revenue_bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }


# ============================================================
# 6. BUSINESS DECISION FRAMEWORK
# ============================================================

def make_recommendation(conversion, revenue):
    """
    Apply the predefined statistical and practical criteria.

    This is a preliminary recommendation, subject to
    experiment-design and guardrail checks.
    """

    significant = conversion["statistically_significant"]
    practical = conversion["practically_significant"]

    revenue_ci_lower = revenue["revenue_ci_lower"]

    if significant and practical and revenue_ci_lower >= 0:
        recommendation = "GO TO ROLLOUT REVIEW"
        explanation = (
            "Conversion lift meets the statistical and practical "
            "thresholds, and the revenue-per-user CI is non-negative. "
            "Verify guardrails and experiment validity before rollout."
        )

    elif significant and conversion["absolute_lift"] < 0:
        recommendation = "NO-GO"
        explanation = (
            "The observed conversion effect is statistically "
            "significantly negative. Investigate before rollout."
        )

    elif significant and not practical:
        recommendation = "ITERATE"
        explanation = (
            "The effect is statistically detectable but does not "
            "meet the predefined minimum practical lift."
        )

    else:
        recommendation = "INCONCLUSIVE"
        explanation = (
            "The evidence does not meet the predefined statistical "
            "criteria. Review power, uncertainty, and guardrails "
            "before deciding whether to continue or redesign."
        )

    return recommendation, explanation


# ============================================================
# 7. SAVE REPORT
# ============================================================

def save_results(metrics, conversion, chi_square, revenue,
                 recommendation, explanation):
    """Save all results for reproducibility."""

    metrics.reset_index().to_csv(
        OUTPUT_DIR / "group_metrics.csv",
        index=False,
    )

    test_results = {
        **conversion,
        **chi_square,
        **revenue,
        "alpha": ALPHA,
        "confidence_level": CONFIDENCE_LEVEL,
        "minimum_practical_lift": MIN_PRACTICAL_LIFT,
        "recommendation": recommendation,
        "recommendation_explanation": explanation,
    }

    pd.DataFrame([test_results]).to_csv(
        OUTPUT_DIR / "ab_test_results.csv",
        index=False,
    )

    return test_results


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 65)
    print("ADVANCED A/B TESTING REPORT")
    print("=" * 65)

    df = load_data()

    metrics = calculate_group_metrics(df)

    conversion = conversion_test(metrics)
    chi_square = chi_square_test(metrics)
    revenue = bootstrap_revenue_test(df)

    recommendation, explanation = make_recommendation(
        conversion,
        revenue,
    )

    print("\nCONVERSION HYPOTHESIS TEST")
    print("-" * 65)

    print(f"Significance level: {ALPHA:.3f}")
    print(
        f"Control conversion: "
        f"{conversion['control_conversion']:.4%}"
    )
    print(
        f"Treatment conversion: "
        f"{conversion['treatment_conversion']:.4%}"
    )
    print(
        f"Absolute lift: "
        f"{conversion['absolute_lift_pp']:.4f} percentage points"
    )
    print(
        f"Relative lift: "
        f"{conversion['relative_lift_pct']:.2f}%"
    )
    print(f"Z-statistic: {conversion['z_statistic']:.4f}")
    print(f"P-value: {conversion['p_value']:.8f}")
    print(
        "95% CI for absolute lift: "
        f"[{conversion['ci_lower_pp']:.4f}, "
        f"{conversion['ci_upper_pp']:.4f}] percentage points"
    )
    print(f"Cohen's h: {conversion['cohens_h']:.4f}")

    print("\nCHI-SQUARE CROSS-CHECK")
    print("-" * 65)
    print(
        f"Chi-square statistic: "
        f"{chi_square['chi_square_statistic']:.4f}"
    )
    print(
        f"Chi-square p-value: "
        f"{chi_square['chi_square_p_value']:.8f}"
    )

    print("\nREVENUE PER USER")
    print("-" * 65)
    print(
        f"Control: "
        f"${revenue['control_revenue_per_user']:.4f}"
    )
    print(
        f"Treatment: "
        f"${revenue['treatment_revenue_per_user']:.4f}"
    )
    print(
        f"Difference: "
        f"${revenue['revenue_per_user_difference']:.4f}"
    )
    print(
        "95% bootstrap CI for difference: "
        f"[${revenue['revenue_ci_lower']:.4f}, "
        f"${revenue['revenue_ci_upper']:.4f}]"
    )

    print("\nBUSINESS DECISION")
    print("-" * 65)
    print(f"Statistically significant: "
          f"{conversion['statistically_significant']}")
    print(f"Practically significant: "
          f"{conversion['practically_significant']}")
    print(f"Recommendation: {recommendation}")
    print(explanation)

    save_results(
        metrics,
        conversion,
        chi_square,
        revenue,
        recommendation,
        explanation,
    )

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
