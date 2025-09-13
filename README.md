# RANK-OPS: Production Recommender with Contextual Bandits & OPE

RANK-OPS is a 20-day sprint project to build a **production-ready recommender system**
that combines:
- Candidate generation (ALS/FAISS)
- Ranking (LightGBM/XGBoost)
- Online learning with contextual bandits (LinUCB / Thompson Sampling)
- Offline Policy Evaluation (IPS/DR/Switch-DR)

Stack: Python 3.11, Poetry, DuckDB/Postgres, FAISS/ALS, LightGBM/XGBoost, FastAPI, GCP Cloud Run, GitHub Actions.

## Roadmap (20 Days)
- Day 1: Kickoff & Repo scaffolding
- Day 2: Data contracts & feature plan
- Day 3: Candidate generation v1
- Day 4: Ranker v1 (baseline)
- Day 5: FastAPI MVP + local inference
- Day 6: Bandit policy (simulator)
- Day 7: OPE v1 (IPS/DR)
- Day 8: Cloud deploy (MVP)
- Day 9: Data & model monitoring
- Day 10: Feature engineering v2 + ranker v2
- Day 11: Bandit v2 + safety rails
- Day 12: Cost, latency, and autoscaling
- Day 13: Retraining & CI
- Day 14: Business case draft + interim demo
- Day 15: A/B design + ethics/fairness
- Day 16: Hardening & incident playbook
- Day 17: Final polish: UX & API ergonomics
- Day 18: Final case study + video
- Day 19: GitHub packaging
- Day 20: Launch

## Day 1

🚀 Kicked off the project repo with clean scaffolding, CI/CD (GitHub Actions), testing, and pre-commit hooks.

**Learning log:**
- ✅What worked: Fast reproducible setup with Poetry + CI from the start.
- ⚠️What broke: Poetry’s default package install needed a src/ package fix.
- ⏭️What’s next: Formalize data contracts and feature plan tomorrow (Day 2).

## Day 2

Added a feature catalog + data contracts for MovieLens-25M. Built a DuckDB + SQL feature pipeline (user/movie/interactions). Validated schema & null safety with pytest + CI. Automated dataset download & reproducibility.

**Learning log:**
- ✅What worked: DuckDB pipelines are blazing fast & reproducible.
- ⚠️What broke: Large dataset download is heavy for CI — mitigated by sampling.
- ⏭️What’s next: Candidate generation v1 (ALS/FAISS).
