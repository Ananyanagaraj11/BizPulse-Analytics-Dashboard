import argparse
import os

import pandas as pd
from sqlalchemy import create_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV into SQLite database.")
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "orders.csv"),
        help="Path to CSV file.",
    )
    parser.add_argument(
        "--db",
        default=os.path.join("data", "orders.db"),
        help="Path to SQLite database file.",
    )
    parser.add_argument(
        "--table",
        default="orders",
        help="Table name to write to.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    engine = create_engine(f"sqlite:///{args.db}")
    df.to_sql(args.table, engine, if_exists="replace", index=False)

    print(f"Wrote {len(df)} rows to {args.db} ({args.table}).")


if __name__ == "__main__":
    main()

