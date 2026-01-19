import argparse
import os

import pandas as pd
from sqlalchemy import create_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV into Postgres database.")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "orders.csv"),
        help="Path to CSV file.",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres SQLAlchemy URL, e.g. postgresql+psycopg2://user:pass@localhost:5432/bizpulse",
    )
    parser.add_argument(
        "--table",
        default="orders",
        help="Table name to write to.",
    )
    args = parser.parse_args()

    if not args.db_url:
        raise SystemExit("Missing --db-url or DATABASE_URL.")

    df = pd.read_csv(args.csv)
    engine = create_engine(args.db_url)
    df.to_sql(args.table, engine, if_exists="replace", index=False)

    print(f"Wrote {len(df)} rows to {args.table} in Postgres.")


if __name__ == "__main__":
    main()

