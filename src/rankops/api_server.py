from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RANK-OPS Recommender")


# Dummy in-memory data
item_ids = list(range(1, 101))
api_key = "secure-api-key"  # TODO: Use a secure key management later


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]


class FeedbackItem(BaseModel):
    user_id: int
    item_id: int
    clicked: bool


@app.get("/")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: int, top_k: int = 10, Authorization: Optional[str] = Header(None)
):
    # API key auth
    if Authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    # TODO: generate actual recommendations later
    import random

    recs = random.sample(item_ids, top_k)
    return RecommendResponse(user_id=user_id, recommendations=recs)


@app.post("/feedback")
def feedback(feedback: FeedbackItem):
    # Log the feedback (clicked or not) for OPE
    print(f"Received feedback: {feedback}")
    return {"status": "ok"}
