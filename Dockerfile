FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml /app/
RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi
COPY . /app
EXPOSE 8080
CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8080"]
