# RANK-OPS

**End-to-end recommender:**
- ALS candidate generation
- LightGBM ranker
- Epsilon-greedy bandit
- OPE (IPS/DR)
- FastAPI service
- Drift monitoring

## Quickstart
```bash
pip install poetry
poetry install
cp .env.example .env
# set API_KEY in .env
```

## End-to-End smoke test (data → train → serve → feedback → OPE → monitoring)
```bash
poetry run python src/smoke.py
```

## End-to-End full run
1) `poetry run python src/data.py` → build `data/ratings.parquet`
2) `poetry run python src/train.py` → writes `models/*.pkl`
3) `poetry run uvicorn src.app:app --reload --port 8080` → open `http://localhost:8080/docs`
4) GET `/recommend?user_id=1` with header `x-api-key: $API_KEY`
5) POST `/feedback` with (user_id,item_id,reward); then run `poetry run python src/ope.py` → view `data/ope_uplift.csv`
6) `poetry run python src/monitor.py` → open `data/evidently_report.html`

## Deploy (Cloud Run quickstart)
gcloud builds submit --config cloudbuild.yaml --project <PROJECT_ID>
