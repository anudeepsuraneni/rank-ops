FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gfortran libopenblas-dev pkg-config git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi || true
COPY . /app
ARG RUN_SMOKE=1
ENV RUN_SMOKE=${RUN_SMOKE}
RUN if [ "$RUN_SMOKE" = "1" ]; then \
    python -m src.data --small || python -m src.data; \
    python -m src.train --small || python -m src.train; \
  fi
EXPOSE 8080
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8080"]
