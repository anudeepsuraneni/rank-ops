import numpy as np
from typing import List, Tuple, Optional
import os, json

def explore(scores: np.ndarray, eps: float = 0.05) -> np.ndarray:
    noise = np.random.rand(len(scores)) * eps
    return scores + noise

class LinUCB:
    def __init__(self, d: int, alpha: float = 1.0, path: str = "models/linucb.json"):
        self.d = d; self.alpha = alpha; self.path = path
        self.A = np.eye(d); self.b = np.zeros((d,1))
        if os.path.exists(path): self.load()

    def score(self, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mu = float(x.T @ theta)
        ci = float(self.alpha * np.sqrt(x.T @ A_inv @ x))
        return mu + ci

    def update(self, x: np.ndarray, r: float):
        x = x.reshape(-1,1)
        self.A += x @ x.T
        self.b += r * x
        self.save()

    def save(self) -> None:
        s = {"A": self.A.tolist(), "b": self.b.tolist(), "d": self.d, "alpha": self.alpha}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(s, open(self.path,"w"))

    def load(self) -> None:
        s = json.load(open(self.path))
        import numpy as np
        self.A = np.array(s["A"], dtype=float)
        self.b = np.array(s["b"], dtype=float)
        self.d = int(s["d"]); self.alpha = float(s["alpha"])

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()

def apply_bandit(scores: np.ndarray,
                 item_ids: List[int],
                 policy: str = "epsilon",
                 user_ctx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if policy == "linucb" and user_ctx is not None:
        lin = LinUCB(d=1, alpha=0.5)
        feats = np.array(scores).reshape(-1,1)     # d=1 feature = ranker_score
        ucb_scores = np.array([lin.score(x) for x in feats])
        probs = softmax(ucb_scores)
        return ucb_scores, probs
    noisy = explore(scores, eps=0.05)
    probs = softmax(noisy)
    return noisy, probs

def apply_safety(items: List[int], recent: List[int]) -> List[int]:
    recent_set = set(recent)
    return [it for it in items if it not in recent_set]
