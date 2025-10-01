# Incident Playbook (RANK-OPS)

**Triggers:**
- Error rate >5% in 5 min
- p95 latency >1s
- Drift score > threshold
- Empty candidates >10%

**Automatic actions:**
- Flip to “incident mode” → fallback to popularity.
- Log WARN + counters.

**Manual runbook:**
- Check Cloud Run logs & /metrics.
- Validate models exist (als.pkl, ranker.pkl, faiss.index).
- If drift: retrain, reduce exploration.
- Once stable 30 min → disable incident mode.

**Postmortem:** What happened, where detected, fixes, learnings.
