# Cloud Run Deploy (one command)

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SERVICE=rank-ops,_REGION=us-central1,_API_KEY=changeme,_CANDIDATE_BACKEND=ALS
```

Then call:
```bash
curl -H "x-api-key: changeme" "$CLOUD_RUN_URL/recommend?user_id=1"
curl -H "x-api-key: changeme" -X POST "$CLOUD_RUN_URL/feedback?user_id=1&item_id=50&reward=1"
curl "$CLOUD_RUN_URL/metrics"
```
