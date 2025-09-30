from src.data import ingest
from src.train import main as train
from src.ope import load_feedback, ips, dr
from src.monitor import main as monitor
from pathlib import Path
import os

DATA = Path("data")
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP = DATA / "ml-latest-small.zip"
EXTRACT = DATA / "ml-latest-small"

def main() -> bool:
    # 1. Ingest
    ingest(URL, ZIP, EXTRACT)

    # 2. Train models
    train()

    # 3. API health & recommend
    from src.app import health, recommend, feedback
    print("RANK-OPS API Health:", health())
    API_KEY = os.getenv("API_KEY")
    recs = recommend(1, API_KEY)
    print("Top-20 Recommendations:", recs)

    # 4. Log some feedback
    print("Feedback:")
    for it in recs["items"][:2]:
        print(feedback(1, int(it), 1, API_KEY))

    # 5. Compute OPE (IPS/DR) from feedback log
    fb = load_feedback()
    if not fb.empty:
        fb["target_prob"] = fb["propensity"]  # demo assumption
        fb["q_hat"] = fb["reward"].rolling(10, min_periods=1).mean()
        print("OPE IPS:", ips(fb), "DR:", dr(fb))
    else:
        print("No feedback yet → OPE skipped")

    # 6. Evidently drift report
    monitor()

    return True

if __name__ == "__main__":
    main()
