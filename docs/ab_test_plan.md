# A/B Test Plan (RANK-OPS)

**Objective:** Validate bandits-enabled ranker improves CTR@K.

## Design
- Random 50/50 split by user_id hash.
- Unit = user-session.
- Primary metric = CTR@K.
- Guardrails = latency p95 <500ms, complaint rate, bounce rate.

## Sample size (approx):
n ≈ 16 × p(1–p) / Δ²
e.g., baseline CTR=6%, uplift Δ=+5% → ~361 users/arm.

## Analysis
- Two-sample z-test for proportions.
- CUPED optional.
- Report effect size + 95% CI.

## Rollout
25% → 50% → 100% if guardrails pass.
