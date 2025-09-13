import duckdb
import pytest
from pathlib import Path
import os


@pytest.fixture(scope="session")
def feature_df():
    con = duckdb.connect(database=":memory:")
    # Use sample in CI, full locally
    if os.getenv("CI") == "true":
        data_path = Path("data/sample/ml-25m/ratings_50k.csv")
    else:
        data_path = Path("data/raw/ml-25m/ratings.csv")
    con.execute(
        f"""
        CREATE TABLE ratings AS
        SELECT * FROM read_csv_auto('{data_path}')
    """
    )
    with open("sql/features.sql", "r") as f:
        con.execute(f.read())
    return con.execute("SELECT * FROM interaction_features LIMIT 1000").df()


def test_no_nulls(feature_df):
    assert not feature_df.isnull().any().any()


def test_schema(feature_df):
    expected_cols = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
        "user_avg_rating",
        "user_rating_count",
        "movie_avg_rating",
        "movie_rating_count",
    }
    assert expected_cols.issubset(set(feature_df.columns))
