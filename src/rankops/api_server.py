from fastapi import FastAPI

app = FastAPI(title="RANK-OPS Recommender")


@app.get("/")
def health_check() -> dict:
    return {"status": "ok"}
