from fastapi import FastAPI

app = FastAPI(title="RANK-OPS Recommender (Day 1)") 

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend(user_id: int, k: int = 10):
    # TODO: load model artifact / index in later days
    return {"user_id": user_id, "recommendations": []}
