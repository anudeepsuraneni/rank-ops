import json, pandas as pd

FEEDBACK_LOG = "data/feedback.log"

def ips(df: pd.DataFrame) -> float:
    w = df["target_prob"] / df["propensity"]
    return float((w * df["reward"]).sum() / w.sum())

def dr(df: pd.DataFrame) -> float:
    w = df["target_prob"] / df["propensity"]
    return float(df["q_hat"].mean() + (w * (df["reward"] - df["q_hat"])).mean())

def load_feedback() -> pd.DataFrame:
    rows = []
    with open(FEEDBACK_LOG) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    fb = load_feedback()
    fb["target_prob"] = fb["propensity"]
    fb["q_hat"] = fb["reward"].rolling(10, min_periods=1).mean()
    out = pd.DataFrame({"IPS":[ips(fb)], "DR":[dr(fb)]})
    out.to_csv("data/ope_uplift.csv", index=False)
    print("Wrote data/ope_uplift.csv")
