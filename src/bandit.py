import numpy as np
from typing import List, Tuple, Optional

def explore(scores: np.ndarray, eps: float = 0.05) -> np.ndarray:
    noise = np.random.rand(len(scores)) * eps
    return scores + noise

class LinUCB:
    def __init__(self, alpha: float = 1.0, dim: int = 1):
        self.alpha = alpha
        self.dim = dim
        self.A = {}
        self.b = {}

    def _init(self, arm: int):
        self.A[arm] = np.eye(self.dim)
        self.b[arm] = np.zeros(self.dim)

    def score(self, arm: int, x: np.ndarray) -> float:
        if arm not in self.A: self._init(arm)
        A_inv = np.linalg.inv(self.A[arm]); theta = A_inv.dot(self.b[arm])
        return float(theta.dot(x) + self.alpha * np.sqrt(x.dot(A_inv).dot(x)))

    def update(self, arm: int, x: np.ndarray, r: float):
        if arm not in self.A: self._init(arm)
        x = x.reshape(-1, 1)
        self.A[arm] += x.dot(x.T)
        self.b[arm] += (r * x.ravel())

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()

def apply_bandit(scores: np.ndarray,
                 item_ids: List[int],
                 policy: str = "epsilon",
                 user_ctx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    if policy == "linucb" and user_ctx is not None:
        lin = LinUCB(alpha=0.5, dim=len(user_ctx))
        ucb_scores = np.array([lin.score(it, user_ctx) for it in item_ids])
        probs = softmax(ucb_scores)
        return ucb_scores, probs
    noisy = explore(scores, eps=0.05)
    probs = softmax(noisy)
    return noisy, probs

def apply_safety(items: List[int], recent: List[int]) -> List[int]:
    recent_set = set(recent)
    return [it for it in items if it not in recent_set]
