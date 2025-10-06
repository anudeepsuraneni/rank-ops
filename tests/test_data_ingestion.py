from rankops.data_ingestion import load_data


def test_load_data_missing(tmp_path) -> None:
    # Create empty directory without ratings.csv
    temp_dir = tmp_path / "data"
    temp_dir.mkdir()
    conn = load_data(str(temp_dir))
    assert conn is None


def test_load_data(tmp_path) -> None:
    # Create fake ratings.csv and movies.csv
    temp_dir = tmp_path / "data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df1 = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [10, 20],
            "rating": [4.0, 3.0],
            "timestamp": [100, 200],
        }
    )
    df2 = pd.DataFrame({"movieId": [10, 20], "title": ["A", "B"]})
    df1.to_csv(temp_dir / "ratings.csv", index=False)
    df2.to_csv(temp_dir / "movies.csv", index=False)
    conn = load_data(str(temp_dir))
    assert conn is not None
    result = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()
    assert result and result[0] == 2
