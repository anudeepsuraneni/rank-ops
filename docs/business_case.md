# Business Case — RANK‑OPS

**Goal:** Lift CTR@K (or Add‑to‑Cart@K) using bandits + counterfactual evaluation (OPE) to safely ship and measure ROI.

## CTR → Revenue model

Let:
- `S` = monthly recommender **sessions** (impressions of slates)
- `CTR` = baseline click‑through rate per session
- `Δ` = relative CTR uplift from new policy (OPE/online)
- `CVR` = conversion rate post‑click
- `AOV` = average order value (USD)
- `g` = gross margin (%)
- `C` = monthly run cost (infra + eng)

Then incremental gross profit per month:
```
ΔGP = S × (CTR × Δ) × CVR × AOV × g − C
```
Example: `S=5M, CTR=6%, Δ=+5%, CVR=7%, AOV=$40, g=55%`
```
ΔGP = 5e6 × (0.06×0.05) × 0.07 × 40 × 0.55 ≈ $18,480 / month
```

## Experiment design (A/B)

- **Primary:** CTR@K (or Add‑to‑Cart@K)
- **Guardrails:** bounce rate, complaint rate, latency p95, null‑click rate
- **Minimum sample size (per arm, two‑sample proportions):**
```
n ≈ 2 * ( z_(1−α/2) * sqrt(p̄(1−p̄)) + z_(1−β) * sqrt(p1(1−p1)+p2(1−p2)) / 2 )^2 / (p2 − p1)^2
```
Where `p1` = baseline CTR, `p2` = target CTR, `p̄=(p1+p2)/2`.
For rough planning at small uplifts (Δ≪1): `n ≈ 16 × p(1−p) / (Δ²)` for 95%/80% power.

## Offline Policy Evaluation (OPE)

We log **per‑item propensities** for each served slate. Estimators:
- **IPS / SNIPS** — unbiased / normalized
- **DR / DR‑SNIPS** — lower variance via a calibrated `q̂ = P(click|x)` (Platt or isotonic)

For slates (click‑any): aggregate propensities as `1 − ∏(1 − p_i)`.

## Risk / Safety

- Safety fallback to **popularity** when latency/error spikes or drift triggered.
- P95 latency budget: `<200ms` model/logic within `<500ms` API budget.
- Circuit breaker: if 5‑min error rate > 5% or drift score > threshold → fallback.
