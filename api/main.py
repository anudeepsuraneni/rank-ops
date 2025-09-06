from fastapi import FastAPI  # web framework for building APIs

app = FastAPI(title="RANK-OPS Recommender (Day 1)")  # create app with title

@app.get("/health")  # health check endpoint
def health():
    return {"status": "ok"}  # simple service status

@app.get("/recommend")  # recommendation endpoint
def recommend(user_id: int, k: int = 10):
    # TODO: load model later for real recommendations
    return {"user_id": user_id, "recommendations": []}  # placeholder response
