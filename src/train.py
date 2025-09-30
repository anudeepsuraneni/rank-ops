import pandas as pd
import os
from pathlib import Path
from src.models import train_als, train_ranker, build_features

STATS_PATH = "models/serve_stats.pkl"

def main() -> None:
    if (os.path.exists("models/als.pkl") and
        os.path.exists("models/ranker.pkl") and
        os.path.exists(STATS_PATH)): 
        return
    df = pd.read_parquet("data/ratings.parquet")
    Path("models").mkdir(exist_ok=True)
    train_als(df, "models/als.pkl")
    feats = build_features(df)
    train_ranker(feats, "models/ranker.pkl", STATS_PATH)
    print("Training complete")

if __name__ == "__main__":
    main()
