# RANK-OPS

A production-ready recommender & ranking system with contextual bandits and offline policy evaluation (OPE).

## Architecture

This project demonstrates:
- Candidate generation (e.g., ALS or FAISS-based retrieval)
- Ranking model (LightGBM)
- Contextual bandits (LinUCB)
- OPE (IPS, DR, Switch-DR)
- Serving via FastAPI
- CI/CD with GitHub Actions and GCP deployment
- Monitoring with Evidently, Cloud Monitoring, and a Streamlit dashboard
- Case study and supporting docs

## Getting Started

1. Clone the repo and set up environment:
  ```bash
  poetry install
  poetry run pre-commit install
  ```
2. Prepare data:
  ```bash
  poetry run python -m rankops.data_ingestion --download --path data/movielens
  ```
3. Train models:
  ```bash
  poetry run python -m rankops.candidate_generation
  poetry run python -m rankops.ranker
  ```
4. Run API:
  ```bash
  poetry run uvicorn rankops.api_server:app --reload
  ```
  