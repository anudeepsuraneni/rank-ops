from src.data import ingest
from src.train import main as train
from src.ope import main as ope 
from src.monitor import main as monitor
import os

def smoke() -> bool:
    # 1. Ingest
    ingest()

    # 2. Train models
    train()

    # 3. API recommend
    from src.app import recommend, feedback, metrics
    API_KEY = os.getenv("API_KEY")
    recs = recommend(1, API_KEY)
    print("Top-20 Recommendations:", recs)

    # 4. Log some feedback
    print("Feedback:")
    for it in recs["items"][:2]:
        print(feedback(1, int(it), 1, API_KEY))
    
    # 5. Validate metrics
    metrics()

    # 6. Compute OPE (IPS/DR) from feedback log
    ope()

    # 7. Evidently drift report
    monitor()

    return True

def test_smoke() -> None:
    assert smoke() == True
