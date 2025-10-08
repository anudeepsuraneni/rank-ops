import lightgbm as lgb
import pandas as pd
from sklearn.metrics import ndcg_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_ranker(data: pd.DataFrame):
    # Assume 'clicked' column as target
    X = data.drop(columns=["clicked"])
    y = data["clicked"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {"objective": "binary", "metric": "auc"}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)  # type: ignore[attr-defined]
    ndcg = ndcg_score(
        y_val.values.reshape(1, -1), y_pred.reshape(1, -1)  # type: ignore[attr-defined]
    )
    y_pred_binary = (y_pred > 0.5).astype(int)  # type: ignore[attr-defined]
    recall = recall_score(y_val, y_pred_binary)
    print(f"Ranker validation AUC: {auc:.4f}")
    print(f"NDCG@5: {ndcg:.4f}, Recall: {recall:.4f}")
    return model


def main() -> None:
    # Example data
    import os

    import numpy as np

    # Use a slightly larger toy dataset so train/val splits are valid.
    rng = np.random.default_rng(42)
    n_rows = 200  # enough rows so validation has multiple samples
    feature1 = rng.random(n_rows)  # floats in [0,1)
    feature2 = rng.integers(1, 10, size=n_rows)  # small integer feature
    click_prob = 0.2 + 0.6 * feature1  # probabilities in [0.2, 0.8)
    clicked = rng.binomial(1, click_prob)
    data = pd.DataFrame(
        {
            "feature1": feature1,
            "feature2": feature2,
            "clicked": clicked,
        }
    )
    # Quick sanity: ensure both classes are present overall
    if len(np.unique(data["clicked"])) < 2:
        raise RuntimeError("Generated dataset has only one class in 'clicked' column.")
    model = train_ranker(data)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/ranker.txt")
    print("Ranker model saved.")


if __name__ == "__main__":
    main()
