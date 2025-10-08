from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RANK-OPS Recommender")


# TODO: add a secure API key auth later

# Dummy in-memory data
item_ids = list(range(1, 101))


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]


class FeedbackItem(BaseModel):
    user_id: int
    item_id: int
    clicked: bool


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int, top_k: int = 10):
    # TODO: generate actual recommendations later
    import random

    # Placeholder: generate recommendations (here: random)
    recs = random.sample(item_ids, top_k)
    return RecommendResponse(user_id=user_id, recommendations=recs)


@app.post("/feedback")
def feedback(feedback: FeedbackItem):
    # TODO: Log the feedback (clicked or not) for OPE
    print(f"Received feedback: {feedback}")
    return {"status": "ok"}


@app.post("/recommend_batch")
def recommend_batch(user_ids: List[int], top_k: int = 5):
    result = {}
    for uid in user_ids:
        result[uid] = recommend(uid, top_k).recommendations
    return result


@app.get("/similar_items")
def similar_items(item_id: int, top_k: int = 5):
    # TODO: Return top_k similar items (placeholder)
    import random

    sim = random.sample(item_ids, top_k)
    return {"item_id": item_id, "similar_items": sim}


@app.get("/cold_start")
def cold_start(top_k: int = 5):
    # Cold start for as new user (e.g., popular items)
    popular_items = list(range(1, top_k + 1))
    return {"recommendations": popular_items}
