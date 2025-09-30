import pandas as pd, requests, zipfile
from pathlib import Path

DATA = Path("data")
URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP = DATA / "ml-25m.zip"
EXTRACT = DATA / "ml-25m"
OUT = DATA / "ratings.parquet"

def ingest(url_path = URL, zip_path = ZIP, extract_path = EXTRACT) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if OUT.exists(): return
    r = requests.get(url_path, stream=True); r.raise_for_status()
    with open(zip_path, "wb") as f:
        for c in r.iter_content(1<<20):
            f.write(c)
    print("Downloaded", zip_path)
    with zipfile.ZipFile(zip_path) as z: z.extractall(DATA)
    print("Extracted", extract_path)
    df = pd.read_csv(extract_path / "ratings.csv")
    df.rename(columns={"userId":"user_id","movieId":"item_id"}, inplace=True)
    df.to_parquet(OUT, index=False)
    print("Wrote", OUT)

if __name__ == "__main__":
    ingest()
