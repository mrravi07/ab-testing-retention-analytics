
from getpass import getpass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


ROOT_DIR = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT_DIR / "sql" / "04_cohort_retention.sql"
OUTPUT_DIR = ROOT_DIR / "reports" / "sql"

COHORT_OUTPUT = OUTPUT_DIR / "weekly_cohort_retention.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "cohort_retention_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not SQL_FILE.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")

    sql_query = SQL_FILE.read_text(encoding="utf-8").strip()
    password = getpass("Enter PostgreSQL password: ")

    connection_url = URL.create(
        "postgresql+psycopg2",
        username="postgres",
        password=password,
        host="localhost",
        port=5432,
        database="ab_testing_db",
    )

    engine = create_engine(connection_url)

    try:
        with engine.connect() as connection:
            result = pd.read_sql_query(
                text(sql_query),
                connection,
            )

        result.to_csv(COHORT_OUTPUT, index=False)

        # Average across eligible signup cohorts, matching the
        # cohort-level summary approach used in the Python analysis.
        summary = (
            result.dropna(subset=["retention_rate_pct"])
            .groupby(["variant", "cohort_age_week"])
            .agg(
                eligible_cohorts=("signup_week", "nunique"),
                average_cohort_retention_pct=(
                    "retention_rate_pct", "mean"
                ),
            )
            .reset_index()
        )

        summary["average_cohort_retention_pct"] = (
            summary["average_cohort_retention_pct"].round(4)
        )

        summary.to_csv(SUMMARY_OUTPUT, index=False)

        print("\nSQL COHORT RETENTION SUMMARY")
        print("=" * 75)
        print(summary.to_string(index=False))

        print("\nCOHORT DETAIL SAMPLE")
        print(result.head(12).to_string(index=False))

        print(f"\nFull cohort data: {COHORT_OUTPUT}")
        print(f"Retention summary: {SUMMARY_OUTPUT}")

        print(f"\nTotal cohort-age rows: {len(result):,}")
        print(
            "Incomplete weeks masked:",
            int(result["retention_rate_pct"].isna().sum()),
        )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

