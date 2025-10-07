#!/bin/bash
GCP_PROJECT="as-rank-ops"
SERVICE_NAME="rank-ops"

echo "Building Docker image..."
docker build -t gcr.io/$GCP_PROJECT/$SERVICE_NAME:latest .

echo "Pushing Docker image..."
docker push gcr.io/$GCP_PROJECT/$SERVICE_NAME:latest

echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$GCP_PROJECT/$SERVICE_NAME:latest \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --project $GCP_PROJECT
