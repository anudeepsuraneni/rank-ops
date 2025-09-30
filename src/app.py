import os, pickle, time, json
from typing import List
from fastapi import FastAPI, Header, HTTPException
from src.models import recommend_als, score_ranker, popular_fallback
from src.bandit import apply_bandit, apply_safety

API_KEY = os.getenv("API_KEY")
TOP_K = 20
CANDIDATE_K = 500
POLICY = "epsilon"
STATS_PATH = "models/serve_stats.pkl"
SERVED_LOG = "data/served.log"
FEEDBACK_LOG = "data/feedback.log"

ALS = pickle.load(
    open("models/als.pkl","rb")) if os.path.exists("models/als.pkl") else None
RANKER = pickle.load(
    open("models/ranker.pkl","rb")) if os.path.exists("models/ranker.pkl") else None

app = FastAPI(title="RANK-OPS API", version="1.0.0")

def check_key(x_api_key: str|None):
    if x_api_key != API_KEY:
        raise HTTPException(401,"bad key")

@app.get("/health")
def health() -> dict:
    return {"ok": True, "als": bool(ALS), "ranker": bool(RANKER)}

@app.get("/recommend")
def recommend(user_id: int, x_api_key: str|None = Header(default=None)) -> dict:
    check_key(x_api_key)
    t0 = time.time()
    if ALS is None or RANKER is None:
        raise HTTPException(503, "models missing")
    cands = recommend_als(ALS, user_id, CANDIDATE_K)
    if not cands:
        cands = popular_fallback(STATS_PATH, CANDIDATE_K)
    recent = _get_recent_items(user_id)
    cands = apply_safety(cands, recent)
    scores = score_ranker(RANKER, cands, STATS_PATH)
    adj_scores, props = apply_bandit(scores, cands, policy=POLICY, user_ctx=None)
    top_idx = adj_scores.argsort()[::-1][:TOP_K]
    items_top = [int(cands[i]) for i in top_idx]
    props_top = [float(props[i]) for i in top_idx]
    _log_served(user_id, items_top, props_top)
    latency_ms = int((time.time() - t0) * 1000)
    return {"user_id": user_id, "items": items_top, "latency_ms": latency_ms}

@app.post("/feedback")
def feedback(user_id: int, item_id: int, reward: int,
             x_api_key: str|None = Header(default=None)) -> dict:
    check_key(x_api_key)
    prop = _lookup_propensity(user_id, item_id)
    _log_feedback(user_id, item_id, reward, prop)
    return {"status": "logged", "propensity": prop}

def _log_served(user_id: int, items: List[int], props: List[float]) -> None:
    os.makedirs(os.path.dirname(SERVED_LOG), exist_ok=True)
    rec = {"ts": int(time.time()), "user_id": user_id, "items": items, "props": props}
    with open(SERVED_LOG, "a") as f: f.write(json.dumps(rec) + "\n")

def _get_recent_items(user_id: int) -> List[int]:
    if not os.path.exists(SERVED_LOG): return []
    try:
        with open(SERVED_LOG, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell()
            f.seek(max(0, size - 4096))
            lines = f.read().decode().splitlines()
        for line in reversed(lines):
            rec = json.loads(line)
            if rec.get("user_id") == user_id:
                return rec.get("items", [])
    except Exception:
        pass
    return []

def _lookup_propensity(user_id: int, item_id: int) -> float:
    if not os.path.exists(SERVED_LOG): return 1.0
    with open(SERVED_LOG) as f:
        for line in reversed(f.readlines()):
            rec = json.loads(line)
            if rec.get("user_id") == user_id:
                items, props = rec["items"], rec["props"]
                for it, pr in zip(items, props):
                    if it == item_id: return float(pr)
                break
    return 1.0

def _log_feedback(user_id: int, item_id: int, reward: int, propensity: float) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    rec = {"ts": int(time.time()), "user_id": user_id, "item_id": item_id,
           "reward": int(reward), "propensity": float(propensity)}
    with open(FEEDBACK_LOG, "a") as f: f.write(json.dumps(rec) + "\n")
