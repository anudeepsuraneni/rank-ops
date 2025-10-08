import numpy as np
import pandas as pd

from rankops.ope import dr_policy_value, ips_policy_value, switch_dr_policy_value


def test_ips_dr_switch() -> None:
    logs = pd.DataFrame({"reward": [1, 0, 1], "propensity": [0.5, 0.5, 1.0]})
    target_probs = np.array([0.4, 0.4, 0.2])
    q_values = pd.Series([0.3, 0.3, 0.7])
    ips = ips_policy_value(logs, target_probs)
    dr = dr_policy_value(logs, target_probs, q_values)
    switch = switch_dr_policy_value(logs, target_probs, q_values, threshold=0.4)
    assert isinstance(ips, float)
    assert isinstance(dr, float)
    assert isinstance(switch, float)
