import json, pandas as pd
import numpy as np

FEEDBACK_LOG='data/feedback.log'; OUT='data/ope_uplift.csv'

def ips(df: pd.DataFrame) -> float:
    w = df['target_prob'] / df['propensity']
    return float((w * df['reward']).sum() / w.sum())

def snips(df: pd.DataFrame) -> float:
    w = df['target_prob'] / df['propensity']
    wn = w / (w.sum() + 1e-12)
    return float((wn * df['reward']).sum())

def dr(df: pd.DataFrame) -> float:
    w = df['target_prob'] / df['propensity']
    return float(df['q_hat'].mean() + (w * (df['reward'] - df['q_hat'])).mean())

def dr_snips(df: pd.DataFrame) -> float:
    w = df['target_prob'] / df['propensity']
    wn = w / (w.sum() + 1e-12)
    return float(df['q_hat'].mean() + (wn * (df['reward'] - df['q_hat'])).sum())

def _prepare_xy(df: pd.DataFrame):
    X = df["score"].to_numpy(dtype=np.float64)
    y = df["reward"].to_numpy(dtype=np.int32)
    m = np.isfinite(X) & np.isfinite(y)
    X, y = X[m], y[m]
    return X, y

def calibrate_platt(df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    X, y = _prepare_xy(df)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if np.unique(y).size < 2:
        const = float(np.clip(y.mean() if y.size else 0.0, 0.0, 1.0))
        q = np.full(shape=(X.shape[0],), fill_value=const, dtype=np.float64)
        return q, None
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X, y)
    q = lr.predict_proba(X)[:, 1].astype(np.float64)
    return q, lr

def calibrate_isotonic(df: pd.DataFrame):
    from sklearn.isotonic import IsotonicRegression
    X, y = _prepare_xy(df)
    if np.unique(y).size < 2:
        const = float(np.clip(y.mean() if y.size else 0.0, 0.0, 1.0))
        q = np.full(shape=(X.shape[0],), fill_value=const, dtype=np.float64)
        return q, None
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    q = ir.fit_transform(X, y).astype(np.float64)
    return q, ir

def slate_ips(slates: pd.DataFrame) -> float:
    import numpy as np
    def agg(p_list) -> float:
        p_list = np.clip(np.array(p_list, dtype=float), 1e-6, 1.0)
        return float(1.0 - np.prod(1.0 - p_list))
    numer, denom = 0.0, 0.0
    for _, row in slates.iterrows():
        p_b = agg(row['propensity_list'])
        p_t = agg(row['target_prob_list'])
        w = p_t / p_b
        numer += w * float(row['reward'])
        denom += w
    return float(numer / (denom + 1e-12))

def load_feedback()->pd.DataFrame:
    rows=[]; 
    try:
        with open(FEEDBACK_LOG) as f:
            for line in f: rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    df = pd.DataFrame(rows)
    if df.empty: return df
    df['score'] = df['propensity']
    return df

def main() -> None:
    fb = load_feedback()
    if fb.empty:
        print('No feedback yet → OPE skipped')
    else:
        q_hat,_ = calibrate_platt(fb)
        fb['q_hat']=q_hat
        fb['target_prob']=fb['propensity']
        out={'IPS':ips(fb),'SNIPS':snips(fb),'DR':dr(fb),'DR_SNIPS':dr_snips(fb),'n':len(fb)}
        pd.DataFrame([out]).to_csv(OUT,index=False); print('Wrote', OUT, out)

if __name__=='__main__':
    main()
