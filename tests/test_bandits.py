import numpy as np

from rankops.bandits import LinUCB


def test_linucb_select_update() -> None:
    n_arms = 3
    n_features = 2
    bandit = LinUCB(n_arms, n_features, alpha=0.5)
    context = np.array([1.0, 2.0])
    arm = bandit.select_arm(context)
    assert 0 <= arm < n_arms
    bandit.update(arm, reward=1.0, user_context=context)
    assert bandit.A[arm][0, 0] == 1.0 + context[0] * context[0]
