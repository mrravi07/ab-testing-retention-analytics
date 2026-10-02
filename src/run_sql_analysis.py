
from getpass import getpass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


ROOT_DIR = Path(__file__).resolve().parents[1]
SQL_FILE = ROOT_DIR / "sql" / "01_experiment_kpis.sql"
OUTPUT_DIR = ROOT_DIR / "reports" / "sql"

OUTPUT_FILE = OUTPUT_DIR / "experiment_kpis.csv"

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

        result.to_csv(OUTPUT_FILE, index=False)

        print("\nSQL EXPERIMENT KPI RESULTS")
        print("=" * 65)
        print(result.to_string(index=False))
        print(f"\nResults exported to: {OUTPUT_FILE}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
