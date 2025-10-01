import pandas as pd, requests, zipfile
from pathlib import Path
import os
import argparse

DATA = Path("data")
OUT = DATA / "ratings.parquet"

def ingest(small = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if os.getenv("CI") == "true" or small:
        url_path = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = DATA / "ml-latest-small.zip"
        extract_path = DATA / "ml-latest-small"
    else:
        url_path = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        zip_path = DATA / "ml-25m.zip"
        extract_path = DATA / "ml-25m"
    if zip_path.exists():
        print(zip_path, "already exists")
    else:
        r = requests.get(url_path, stream=True); r.raise_for_status()
        with open(zip_path, "wb") as f:
            for c in r.iter_content(1<<20):
                f.write(c)
        print("Downloaded", zip_path)
    if extract_path.exists():
        print(extract_path, "already exists")
    else:
        with zipfile.ZipFile(zip_path) as z: z.extractall(DATA)
        print("Extracted", extract_path)
    if OUT.exists():
        print(OUT, "already exists")
    else:
        df = pd.read_csv(extract_path / "ratings.csv")
        df.rename(columns={"userId":"user_id","movieId":"item_id"}, inplace=True)
        df.to_parquet(OUT, index=False)
        print("Wrote", OUT)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="Use small MovieLens dataset")
    args = ap.parse_args()
    ingest(small=args.small)

