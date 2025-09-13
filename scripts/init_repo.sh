#!/usr/bin/env bash
set -euo pipefail

# ---- basic checks ----
if ! command -v poetry >/dev/null 2>&1; then
  echo "ERROR: 'poetry' not found. Install with 'pip install poetry' or see https://python-poetry.org/docs/#installation"
  exit 1
fi

# ---- init repo ----
mkdir -p rank-ops
cd rank-ops

if [ ! -d .git ]; then
  git init
fi

# set repo-local git identity if not configured globally (non-intrusive)
if [ -z "$(git config --get user.name || true)" ]; then
  git config user.name "Anudeep Suraneni"
fi
if [ -z "$(git config --get user.email || true)" ]; then
  git config user.email "anudeepsuraneni@gmail.com"
fi

# Create README
cat > README.md <<'EOF'
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
EOF

# ---- package layout ----
mkdir -p src/rank_ops
cat > src/rank_ops/__init__.py <<'EOF'
# minimal package marker — extend later
__version__ = "0.1.0"
EOF

# ---- pyproject ----
cat > pyproject.toml <<'EOF'
[tool.poetry]
name = "rank-ops"
version = "0.1.0"
description = "Production-ready recommender with contextual bandits & OPE"
authors = ["Anudeep Suraneni <anudeepsuraneni@gmail.com>"]
readme = "README.md"
packages = [
  { include = "rank_ops", from = "src" }
]

[tool.poetry.dependencies]
python = ">=3.11, <3.12"
numpy = "^1.26"
pandas = "^2.1"
scikit-learn = "^1.4"
lightgbm = "^4.3"
xgboost = "^2.0"
duckdb = "^0.10"
faiss-cpu = "^1.8"
fastapi = "^0.111"
uvicorn = "^0.30"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3"
pre-commit = "^3.7"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
EOF

# ---- install deps ----
poetry install

# ---- pre-commit config ----
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 7.1.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.1
    hooks:
      - id: mypy
EOF

poetry run pre-commit install

# ---- CI: run pre-commit + tests ----
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'EOF'
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install Poetry
        run: pip install poetry
      - name: Install deps
        run: poetry install
      - name: Run pre-commit checks
        run: poetry run pre-commit run --all-files
      - name: Run tests
        run: poetry run pytest -q
EOF

# ---- .gitignore ----
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.so
*.egg-info/
*.egg
dist/
build/

# Virtual env
.env
.venv
*.venv/

# IDEs/editors
.vscode/
.idea/

# Data & outputs
data/raw/
outputs/
logs/
checkpoints/

# OS
.DS_Store
Thumbs.db
EOF

# ---- tests ----
mkdir -p tests
cat > tests/test_sanity.py <<'EOF'
def test_sanity():
    assert 1 + 1 == 2
EOF

# ---- license ----
cat > LICENSE <<'EOF'
MIT License

Copyright (c) 2025 Anudeep Suraneni

Permission is hereby granted, free of charge, to any person obtaining a copy...
EOF
