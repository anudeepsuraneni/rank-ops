import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
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
    print(f"Ranker validation AUC: {auc:.4f}")
    return model


def main() -> None:
    # Example data
    data = pd.DataFrame(
        {
            "feature1": [0.1, 0.4, 0.5, 0.7],
            "feature2": [1, 3, 5, 7],
            "clicked": [0, 1, 0, 1],
        }
    )
    model = train_ranker(data)
    model.save_model("models/ranker.txt")
    print("Ranker model saved.")


if __name__ == "__main__":
    main()
