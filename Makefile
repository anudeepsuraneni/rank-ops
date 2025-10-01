.PHONY: setup run train smoke test

setup:
	pip install poetry && poetry config virtualenvs.create false && poetry install

run:
	uvicorn src.app:app --host 0.0.0.0 --port 8080

train:
	python src/data.py && python src/train.py

smoke:
	python src/smoke.py

test:
	pytest -q --cov=src --cov-report=term-missing
