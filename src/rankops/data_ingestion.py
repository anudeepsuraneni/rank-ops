import argparse
import os

import duckdb
import pandas as pd

from rankops.sql_utils import (
    create_session_table,
    create_user_metrics,
    create_user_recency,
)


def download_movielens(dest_path: str):
    """Download and extract MovieLens 25M dataset"""
    os.makedirs(dest_path, exist_ok=True)
    # Placeholder for download code
    print(f"Downloading MovieLens 25M to {dest_path} (simulated)")


def load_data(dest_path: str):
    """Load MovieLens data into DuckDB and return connection"""
    conn = duckdb.connect(database=":memory:")
    ratings_csv = os.path.join(dest_path, "ratings.csv")
    movies_csv = os.path.join(dest_path, "movies.csv")
    if not os.path.exists(ratings_csv):
        print("Ratings file not found. Please download dataset.")
        return None
    if not os.path.exists(movies_csv):
        print("Movies file not found. Please download dataset.")
        return None
    ratings = pd.read_csv(ratings_csv)
    movies = pd.read_csv(movies_csv)
    conn.register("ratings", ratings)
    conn.register("movies", movies)
    # Build derived tables
    create_session_table(conn)
    create_user_metrics(conn)
    create_user_recency(conn)
    return conn


def main():
    parser = argparse.ArgumentParser(description="Data Ingestion")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument(
        "--path", type=str, default="data/movielens", help="Destination path"
    )
    args = parser.parse_args()
    if args.download:
        download_movielens(args.path)
    # Load data
    conn = load_data(args.path)
    if conn:
        print("Data ingestion complete.")


if __name__ == "__main__":
    main()
