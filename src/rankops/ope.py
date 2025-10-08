import numpy as np


def ips_policy_value(logs, target_probs) -> np.float64:
    """Inverse Propensity Score estimator"""
    importance = target_probs / logs["propensity"]
    return np.mean(importance * logs["reward"])


def dr_policy_value(logs, target_probs, q_values) -> np.float64:
    """Doubly Robust estimator"""
    importance = target_probs / logs["propensity"]
    dr = q_values + importance * (logs["reward"] - q_values)
    return np.mean(dr)


def switch_dr_policy_value(logs, target_probs, q_values, threshold=0.1) -> np.float64:
    """Switch Doubly Robust: use IPS when propensities > threshold, else DM"""
    importance = target_probs / logs["propensity"]
    is_large = logs["propensity"] > threshold
    value = []
    for i in range(len(logs)):
        if is_large.iloc[i]:
            value.append(importance.iloc[i] * logs["reward"].iloc[i])
        else:
            value.append(q_values.iloc[i])
    return np.mean(value)


def main() -> None:
    import pandas as pd

    # Simulated logged data
    logs = pd.DataFrame(
        {
            "reward": [1, 0, 1, 0],
            "propensity": [0.2, 0.8, 0.5, 0.3],
            "action": [0, 1, 0, 2],
        }
    )
    # Suppose target_probs computed elsewhere
    target_probs = np.array([0.5, 0.4, 0.1, 0.2])
    q_values = pd.Series([0.6, 0.4, 0.6, 0.3])
    print("IPS:", ips_policy_value(logs, target_probs))
    print("DR:", dr_policy_value(logs, target_probs, q_values))
    print("Switch-DR:", switch_dr_policy_value(logs, target_probs, q_values))


if __name__ == "__main__":
    main()
