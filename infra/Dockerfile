# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install pip and dependencies
RUN pip install --upgrade pip
RUN pip install poetry

# Install Python dependencies via Poetry
RUN poetry config virtualenvs.create false && poetry install

# Expose port for FastAPI
EXPOSE 8000 8080

# Command to run the API
CMD ["sh", "-c", "uvicorn src.rankops.api_server:app --host 0.0.0.0 --port ${PORT:-8080}"]
