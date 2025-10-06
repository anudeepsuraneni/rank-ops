import argparse
import os
import zipfile

import duckdb
import pandas as pd
import requests


def download_movielens(dest_path: str):
    """Download and extract MovieLens 25M dataset"""
    os.makedirs(dest_path, exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_path = os.path.join(dest_path, "ml-25m.zip")
    # Download the dataset
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB
                if chunk:
                    f.write(chunk)
    # Extract the dataset
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_path)
    print("Downloaded MovieLens 25M dataset.")
    # Remove the zip file
    os.remove(zip_path)


def load_data(dest_path: str):
    """Load MovieLens data into DuckDB and return connection"""
    conn = duckdb.connect(database=":memory:")
    ratings_path = os.path.join(dest_path, "ratings.csv")
    movies_path = os.path.join(dest_path, "movies.csv")
    if not os.path.exists(ratings_path):
        print("Ratings file not found. Please download dataset.")
        return None
    if not os.path.exists(movies_path):
        print("Movies file not found. Please download dataset.")
        return None
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    conn.register("ratings", ratings)
    conn.register("movies", movies)
    return conn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Ingestion")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument(
        "--path", type=str, default="data/movielens", help="Destination path"
    )
    args = parser.parse_args()
    if args.download:
        download_movielens(args.path)
    # Load data
    dest_path = os.path.join(args.path, "ml-25m")
    conn = load_data(dest_path)
    if conn:
        print("Data ingestion complete.")
