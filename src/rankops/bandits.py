import numpy as np


class LinUCB:
    def __init__(self, n_arms, n_features, alpha=1.0) -> None:
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        # Initialize A = identity, b = zero for each arm
        self.A = np.array([np.eye(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))

    def select_arm(self, user_context) -> np.int64:
        # user_context shape: (n_features,)
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv.dot(self.b[a])
            p[a] = theta.dot(user_context) + self.alpha * np.sqrt(
                user_context.dot(A_inv).dot(user_context)
            )
        return np.argmax(p)

    def update(self, chosen_arm, reward, user_context) -> None:
        x = user_context
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x


def main() -> None:
    # Simulated bandit interaction
    n_arms = 5
    n_features = 3
    bandit = LinUCB(n_arms, n_features, alpha=0.1)
    user_context = np.array([0.5, 0.2, 0.3])
    arm = bandit.select_arm(user_context)
    print(f"Selected arm: {arm}")
    # Suppose reward obtained is 1
    bandit.update(arm, reward=1, user_context=user_context)


if __name__ == "__main__":
    main()
