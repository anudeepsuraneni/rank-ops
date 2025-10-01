import numpy as np
import pandas as pd
from src.ope import ips, snips, dr, dr_snips, calibrate_platt
from typing import Tuple

def simulate(n=20000, seed=42) -> Tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n,1))
    true_p = 1 / (1 + np.exp(-2*x[:,0]))
    score_b = x[:,0] + rng.normal(0, 0.5, size=n)
    prop = 1 / (1 + np.exp(-score_b))
    prop = np.clip(prop, 0.05, 0.95)
    score_t = score_b + 0.3
    targ = 1 / (1 + np.exp(-score_t))
    r = rng.binomial(1, true_p)
    df = pd.DataFrame({
        "reward": r,
        "propensity": prop,
        "target_prob": targ,
        "score": score_b
    })
    return df, true_p.mean()

def test_estimators_close_to_truth() -> None:
    df, p_true = simulate()
    q_hat, _ = calibrate_platt(df)
    df["q_hat"] = q_hat
    est_ips = ips(df)
    est_snips = snips(df)
    est_dr = dr(df)
    est_drs = dr_snips(df)
    for est in [est_ips, est_snips, est_dr, est_drs]:
        assert 0.0 <= est <= 1.0
    assert abs(est_dr - p_true) < 0.03
