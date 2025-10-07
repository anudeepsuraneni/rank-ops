import faiss
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def build_item_embeddings(ratings: pd.DataFrame, n_components: int = 50) -> np.ndarray:
    # Create user-item rating matrix
    user_item = ratings.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )
    # Ensure n_components ≤ number of items (features after transpose) and ≥ 1
    n_components = max(1, int(min(n_components, user_item.shape[0])))
    # Compute SVD embeddings
    svd = TruncatedSVD(n_components=n_components)
    item_embeddings = svd.fit_transform(user_item.T)
    return item_embeddings


def build_faiss_index(item_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # Build FAISS index for item embeddings
    d = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_embeddings.astype("float32"))  # type: ignore[attr-defined]
    return index


def get_top_n(
    index: faiss.IndexFlatIP,
    item_embeddings: np.ndarray,
    item_ids: list,
    user_profile_vector: np.ndarray,
    n: int = 10,
) -> list:
    # Perform ANN search for top-n items
    distances, indices = index.search(
        user_profile_vector.astype("float32"), n
    )  # type: ignore[attr-defined]
    recommendations = [item_ids[i] for i in indices[0]]
    return recommendations


def main() -> None:
    # Example usage: generate item embeddings and FAISS index
    ratings = pd.DataFrame(
        {"userId": [1, 1, 2, 2], "movieId": [10, 20, 10, 30], "rating": [4, 5, 3, 2]}
    )
    item_embeddings = build_item_embeddings(ratings)
    index = build_faiss_index(item_embeddings)
    print("FAISS index built for item embeddings")
    # Simulated user profile (mean of rated item embeddings)
    user_profile = np.mean(item_embeddings, axis=0).reshape(1, -1)
    item_ids = [10, 20, 30, 40][: item_embeddings.shape[0]]
    recs = get_top_n(index, item_embeddings, item_ids, user_profile, n=3)
    print(f"Top recommendations: {recs}")


if __name__ == "__main__":
    main()
