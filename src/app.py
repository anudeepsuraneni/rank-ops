import os, pickle, time, json, numpy as np
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi import Request, Response
from src.models import score_ranker, popular_fallback
from src.bandit import apply_bandit, apply_safety, LinUCB
from src.candidates import get_candidates
from src.metrics import metrics_endpoint, metrics_middleware, EXPLORE_RATE, CTR
from fastapi.middleware.cors import CORSMiddleware

API_KEY = os.getenv("API_KEY")
TOP_K = int(os.getenv("TOP_K", "20"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "500"))
POLICY = os.getenv("POLICY", "epsilon")
BACKEND = os.getenv("CANDIDATE_BACKEND", "ALS")
DRIFT_FLAG = os.getenv("DRIFT_FLAG_FILE", "data/drift.flag")

STATS_PATH = "models/serve_stats.pkl"
SERVED_LOG = "data/served.log"
FEEDBACK_LOG = "data/feedback.log"
LINUCB_PATH = "models/linucb.json"
LINUCB_ALPHA = 1.0

ALS = pickle.load(open("models/als.pkl","rb")) if os.path.exists("models/als.pkl") else None
RANKER = pickle.load(
    open("models/ranker.pkl","rb")) if os.path.exists("models/ranker.pkl") else None

app = FastAPI(title="RANK-OPS")

ALLOWED = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def _metrics_mw(request: Request, call_next):
    return await metrics_middleware(request, call_next)

@app.get("/metrics")
def metrics() -> Response:
    return metrics_endpoint()

def check_key(x_api_key: Optional[str]) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _incident_mode() -> bool:
    if not (ALS and RANKER): return True
    return os.path.exists(DRIFT_FLAG)

@app.get("/recommend")
def recommend(user_id: int, x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    t0 = time.time()
    recent = _get_recent_items(user_id)
    try:
        cands = get_candidates(user_id, CANDIDATE_K, backend=BACKEND, als_obj=ALS, recent_items=recent)
    except Exception:
        cands = []
    if not cands and not _incident_mode():
        try:
            cands = get_candidates(user_id, CANDIDATE_K, backend="ALS", als_obj=ALS, recent_items=recent)
        except Exception:
            cands = []
    if not cands:
        items_top = popular_fallback(STATS_PATH, TOP_K)        
        _log_served(user_id, items_top, [1.0/len(items_top)]*len(items_top))
        return {"user_id": user_id, "items": items_top, "latency_ms": int((time.time()-t0)*1000), "fallback": True}
    cands = apply_safety(cands, recent)
    scores = score_ranker(RANKER, cands, STATS_PATH)
    adj_scores, props = apply_bandit(scores, cands, policy=POLICY, user_ctx=None)
    EXPLORE_RATE.observe(float((props > (props.mean())).mean()))
    top_idx = adj_scores.argsort()[::-1][:TOP_K]
    items_top = [int(cands[i]) for i in top_idx]
    props_top = [float(props[i]) for i in top_idx]
    scores_top = [float(scores[i]) for i in top_idx]
    _log_served(user_id, items_top, props_top, scores_top)
    latency_ms = int((time.time() - t0) * 1000)
    return {"user_id": user_id, "items": items_top, "latency_ms": latency_ms, "backend": BACKEND}

@app.post("/feedback")
def feedback(user_id: int, item_id: int, reward: int,
             x_api_key: Optional[str] = Header(default=None)) -> dict:
    check_key(x_api_key)
    prop = _lookup_propensity(user_id, item_id)
    _log_feedback(user_id, item_id, reward, prop)
    x = _lookup_feature(user_id, item_id, default_feature=prop)
    lin = LinUCB(d=x.shape[0], alpha=LINUCB_ALPHA, path=LINUCB_PATH)
    lin.update(x, float(reward))
    try:
        CTR.observe(float(reward))
    except Exception:
        pass
    return {"status": "logged", "propensity": prop}

def _log_served(user_id: int, items: list[int], props: list[float],
                scores: list[float] | None = None) -> None:
    os.makedirs(os.path.dirname(SERVED_LOG), exist_ok=True)
    if scores is None:
        scores = [float(p) for p in props]
    rec = {
        "ts": int(time.time()),
        "user_id": user_id,
        "items": items,
        "props": props,
        "scores": scores
    }
    with open(SERVED_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

def _lookup_feature(user_id: int, item_id: int, default_feature: float = 1.0) -> np.ndarray:
    if not os.path.exists(SERVED_LOG):
        return np.array([default_feature], dtype=float)
    try:
        with open(SERVED_LOG, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell()
            f.seek(max(0, size - 131072))
            lines = f.read().decode().splitlines()
        for line in reversed(lines):
            rec = json.loads(line)
            if rec.get("user_id") != user_id: 
                continue
            items = rec.get("items", [])
            if not items: 
                continue
            try:
                idx = [int(x) for x in items].index(int(item_id))
            except ValueError:
                continue
            scores = rec.get("scores") or rec.get("props") or []
            if scores and 0 <= idx < len(scores):
                return np.array([float(scores[idx])], dtype=float)
            props = rec.get("props") or []
            if props and 0 <= idx < len(props):
                return np.array([float(props[idx])], dtype=float)
            break
    except Exception:
        pass
    return np.array([default_feature], dtype=float)

def _get_recent_items(user_id: int) -> List[int]:
    if not os.path.exists(SERVED_LOG): return []
    try:
        with open(SERVED_LOG, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell()
            f.seek(max(0, size - 8192))
            lines = f.read().decode().splitlines()
            for line in reversed(lines):
                rec = json.loads(line)
                if rec.get("user_id") == user_id:
                    return [int(x) for x in rec.get("items", [])]
    except Exception:
        return []
    return []

def _lookup_propensity(user_id: int, item_id: int) -> float:
    if not os.path.exists(SERVED_LOG): return 1.0
    with open(SERVED_LOG) as f:
        for line in reversed(f.readlines()):
            rec = json.loads(line)
            if rec.get("user_id") == user_id:
                for it, pr in zip(rec.get("items", []), rec.get("props", [])):
                    if int(it) == int(item_id):
                        return float(pr)
                break
    return 1.0

def _log_feedback(user_id: int, item_id: int, reward: int, propensity: float) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    rec = {"ts": int(time.time()), "user_id": user_id, "item_id": int(item_id),
           "reward": int(reward), "propensity": float(propensity)}
    with open(FEEDBACK_LOG, "a") as f: f.write(json.dumps(rec) + "\n")
