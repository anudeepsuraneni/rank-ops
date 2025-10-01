import pandas as pd
import os
from pathlib import Path
from src.models import train_als, train_ranker, build_features, build_covis, build_faiss

BACKEND = os.getenv("CANDIDATE_BACKEND", "ALS")

ALS = "models/als.pkl"
RANKER = "models/ranker.pkl"
STATS_PATH = "models/serve_stats.pkl"
COVIS_PATH = "models/covis.pkl"
FAISS_PATH = "models/faiss.index"

def main() -> None:
    df = pd.read_parquet("data/ratings.parquet")
    Path("models").mkdir(exist_ok=True)
    if not os.path.exists(ALS):
        train_als(df, ALS)    
    if not os.path.exists(FAISS_PATH):
        build_faiss(FAISS_PATH)
    if not os.path.exists(COVIS_PATH):
        build_covis(df, COVIS_PATH)
    if not (os.path.exists(RANKER) and os.path.exists(STATS_PATH)):        
        feats = build_features(df)
        train_ranker(feats, RANKER, STATS_PATH)
    print("Training complete")

if __name__ == "__main__":
    main()
