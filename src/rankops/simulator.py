import numpy as np
import pandas as pd


def simulate_interactions(n: int = 1000, n_arms: int = 5):
    """Simulate logged bandit data with propensities and rewards"""
    data = []
    contexts = np.random.rand(n, 3)
    for _ in range(n):
        propensities = np.random.dirichlet(np.ones(n_arms), size=1)[0]
        action = np.random.choice(n_arms, p=propensities)
        reward = np.random.binomial(1, 0.5)  # random reward
        data.append(
            {"action": action, "reward": reward, "propensity": propensities[action]}
        )
    df = pd.DataFrame(data)
    context_df = pd.DataFrame(
        contexts, columns=[f"ctx_{j}" for j in range(contexts.shape[1])]
    )
    logs = pd.concat([df, context_df], axis=1)
    return logs
