
from pathlib import Path
from getpass import getpass

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "ab_testing_db"
DB_USER = "postgres"
SCHEMA = "analytics"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def connect_database():
    password = getpass("Enter PostgreSQL password: ")

    connection_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=password,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

    engine = create_engine(connection_url)

    # Verify the connection.
    with engine.connect() as connection:
        database = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        print(f"\nConnected to PostgreSQL database: {database}")

    return engine


# ============================================================
# CREATE SCHEMA AND TABLES
# ============================================================

def create_tables(engine):
    ddl_statements = [
        f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}",

        f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.users (
            user_id TEXT PRIMARY KEY,
            signup_date DATE NOT NULL,
            country TEXT,
            device TEXT,
            acquisition_channel TEXT
        )
        """,

        f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.experiment_assignments (
            user_id TEXT PRIMARY KEY
                REFERENCES {SCHEMA}.users(user_id),
            experiment_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            assigned_at DATE NOT NULL,
            CHECK (variant IN ('Control', 'Treatment'))
        )
        """,

        f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.events (
            event_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL
                REFERENCES {SCHEMA}.users(user_id),
            session_id TEXT NOT NULL,
            event_name TEXT NOT NULL,
            event_timestamp TIMESTAMP NOT NULL
        )
        """,

        f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}.orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL
                REFERENCES {SCHEMA}.users(user_id),
            order_date TIMESTAMP NOT NULL,
            revenue NUMERIC(12, 2) NOT NULL
                CHECK (revenue >= 0)
        )
        """,
    ]

    with engine.begin() as connection:
        for statement in ddl_statements:
            connection.execute(text(statement))

    print(f"Schema '{SCHEMA}' and tables are ready.")


# ============================================================
# IMPORT CSV FILES
# ============================================================

def import_csv(engine, filename, table_name, date_columns):
    csv_path = RAW_DIR / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="raise")

    # Avoid duplicate inserts if the script is run again.
    with engine.connect() as connection:
        existing_count = connection.execute(
            text(
                f"SELECT COUNT(*) FROM {SCHEMA}.{table_name}"
            )
        ).scalar_one()

    if existing_count > 0:
        print(
            f"SKIPPED {table_name}: already contains "
            f"{existing_count:,} rows."
        )
        return

    df.to_sql(
        name=table_name,
        con=engine,
        schema=SCHEMA,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi",
    )

    print(f"Imported {len(df):,} rows into {SCHEMA}.{table_name}")


# ============================================================
# VALIDATE IMPORT
# ============================================================

def validate_import(engine):
    expected_counts = {
        "users": 50000,
        "experiment_assignments": 50000,
        "events": 227150,
        "orders": 3752,
    }

    print("\n" + "=" * 65)
    print("POSTGRESQL IMPORT VALIDATION")
    print("=" * 65)

    all_passed = True

    with engine.connect() as connection:
        for table_name, expected in expected_counts.items():
            actual = connection.execute(
                text(
                    f"SELECT COUNT(*) FROM {SCHEMA}.{table_name}"
                )
            ).scalar_one()

            passed = actual == expected
            all_passed = all_passed and passed

            status = "PASS" if passed else "CHECK"

            print(
                f"{table_name:<25} "
                f"Rows: {actual:>8,} | "
                f"Expected: {expected:>8,} | {status}"
            )

        invalid_assignments = connection.execute(
            text(f"""
                SELECT COUNT(*)
                FROM {SCHEMA}.users u
                LEFT JOIN {SCHEMA}.experiment_assignments a
                    ON u.user_id = a.user_id
                WHERE a.user_id IS NULL
            """)
        ).scalar_one()

        invalid_event_users = connection.execute(
            text(f"""
                SELECT COUNT(*)
                FROM {SCHEMA}.events e
                LEFT JOIN {SCHEMA}.users u
                    ON e.user_id = u.user_id
                WHERE u.user_id IS NULL
            """)
        ).scalar_one()

        print(f"\nUsers missing assignment: {invalid_assignments}")
        print(f"Events with unknown users: {invalid_event_users}")

        if invalid_assignments != 0 or invalid_event_users != 0:
            all_passed = False

    if all_passed:
        print("\nPASS: PostgreSQL data validation completed.")
    else:
        print(
            "\nWARNING: Check the row counts and relationships "
            "before proceeding."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("LOAD A/B TESTING DATA INTO POSTGRESQL")
    print("=" * 65)

    engine = connect_database()

    try:
        create_tables(engine)

        import_csv(
            engine,
            "users.csv",
            "users",
            ["signup_date"],
        )

        import_csv(
            engine,
            "experiment_assignments.csv",
            "experiment_assignments",
            ["assigned_at"],
        )

        import_csv(
            engine,
            "events.csv",
            "events",
            ["event_timestamp"],
        )

        import_csv(
            engine,
            "orders.csv",
            "orders",
            ["order_date"],
        )

        validate_import(engine)

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()

