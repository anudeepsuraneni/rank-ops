import pickle, os, pandas as pd, numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from typing import cast
from scipy.sparse import csr_matrix

def _compute_item_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["item_popularity"] = df.groupby("item_id").size().astype(int).to_dict()
    stats["item_avg_rating"] = df.groupby("item_id")["rating"].mean().to_dict()
    stats["global_pop_items"] = (
        pd.Series(stats["item_popularity"]).sort_values(ascending=False).index.tolist()
    )
    stats["global_avg_rating"] = float(df["rating"].mean())
    return stats

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["click"] = (out["rating"] >= 4).astype(int)
    item_popularity = df.groupby("item_id").size().rename("item_popularity")
    item_avg_rating = df.groupby("item_id")["rating"].mean().rename("item_avg_rating")
    out = out.merge(item_popularity, on="item_id", how="left")
    out = out.merge(item_avg_rating, on="item_id", how="left")
    out["item_popularity"] = out["item_popularity"].fillna(0).astype(int)
    out["item_avg_rating"] = out["item_avg_rating"].fillna(df["rating"].mean())
    return out

def train_als(df: pd.DataFrame, path: str) -> dict:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1, "blas")
    except Exception:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    import implicit
    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")
    u = users.cat.codes.values; i = items.cat.codes.values
    conf = sparse.coo_matrix((df["rating"].values, (u, i))).tocsr()
    model = implicit.als.AlternatingLeastSquares(factors=32, iterations=10)
    model.fit(conf.T.tocsr())
    obj = {"model": model, "users": list(users.cat.categories),
           "items": list(items.cat.categories)}
    with open(path, "wb") as f: pickle.dump(obj, f)
    print("Saved", path)
    return obj

def train_ranker(feat_df: pd.DataFrame, path: str, stats_path: str) -> dict:
    feats = ["rating", "item_avg_rating", "item_popularity"]
    X, y = feat_df[feats], feat_df["click"]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=3,
        num_leaves=15,
        min_data_in_leaf=20,
        learning_rate=0.1,
        verbosity=-1
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc")
    auc = roc_auc_score(yva, cast(np.ndarray, model.predict_proba(Xva))[:, 1])
    with open(path, "wb") as f: pickle.dump({"model": model, "features": feats}, f)
    stats = _compute_item_stats(feat_df[["item_id", "rating"]])
    with open(stats_path, "wb") as f: pickle.dump(stats, f)
    print("Ranker AUC", auc, "| Saved", path, "and", stats_path)
    return {"model": model, "features": feats}

def _serve_features(item_ids: list[int], stats: dict, feature_order: list[str]) -> pd.DataFrame:
    item_avg = stats.get("item_avg_rating", {})
    global_avg = float(stats.get("global_avg_rating", 3.5))
    pop = stats.get("item_popularity", {})
    item_ids = [int(it) for it in item_ids]
    rating_proxy = [float(item_avg.get(it, global_avg)) for it in item_ids]
    item_avg_list = [float(item_avg.get(it, global_avg)) for it in item_ids]
    pop_list = [int(pop.get(it, 0)) for it in item_ids]
    df = pd.DataFrame({
        "rating": rating_proxy,
        "item_avg_rating": item_avg_list,
        "item_popularity": pop_list,
    })
    return df[feature_order]

def recommend_als(als_obj: dict, user_id: int, k: int) -> list[int]:
    model = als_obj["model"]; users = als_obj["users"]; items = als_obj["items"]
    if user_id not in users:
        return []
    uid = users.index(user_id)
    from scipy.sparse import csr_matrix
    recs = model.recommend(uid, csr_matrix((1, len(items))), N=k)
    return list(recs[0])

def score_ranker(ranker_obj, item_ids: list[int], stats_path: str) -> np.ndarray:
    with open(stats_path, "rb") as f: stats = pickle.load(f)
    feature_order = ranker_obj.get("features", ["rating"])
    X = _serve_features(item_ids, stats, feature_order)
    return ranker_obj["model"].predict_proba(X)[:, 1]

def popular_fallback(stats_path: str, k: int) -> list[int]:
    with open(stats_path, "rb") as f: stats = pickle.load(f)
    return stats["global_pop_items"][:k]

def build_covis(df: pd.DataFrame, path: str, topk: int = 100) -> None:
    users = df['user_id'].astype('int64').unique()
    items = df['item_id'].astype('int64').unique()
    u2i = {u:i for i,u in enumerate(users)}
    it2i = {it:i for i,it in enumerate(items)}
    I2it = {i:it for it,i in it2i.items()}
    dedup = df.drop_duplicates(['user_id','item_id'])
    rows = dedup['user_id'].map(u2i).values
    cols = dedup['item_id'].map(it2i).values
    data = np.ones(len(dedup), dtype=np.float32)
    X = csr_matrix((data,(rows,cols)), shape=(len(users), len(items)), dtype=np.float32)
    C = X.T @ X
    C.setdiag(0); C.eliminate_zeros()
    covis = {}
    C = C.tocsr()
    for i in range(C.shape[0]):
        start,end = C.indptr[i], C.indptr[i+1]
        js = C.indices[start:end]; vs=C.data[start:end]
        if len(js)==0: covis[int(I2it[i])]=[]; continue
        if len(js)>topk:
            idx = np.argpartition(vs, -topk)[-topk:]
            js,vs = js[idx], vs[idx]
        order = np.argsort(-vs)
        covis[int(I2it[i])] = [int(I2it[j]) for j in js[order]]
    with open(path,'wb') as f: pickle.dump(covis,f)
    print(f"Wrote {path} with {len(covis)} items")

def build_faiss(path: str) -> None:
    import faiss
    als = pickle.load(open("models/als.pkl","rb"))
    vecs = np.ascontiguousarray(als["model"].item_factors.astype(np.float32))
    dim = vecs.shape[1]
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100, faiss.METRIC_L2)
    index.train(vecs); index.add(vecs) # type: ignore[reportCallIssue]
    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, path)
    print("Wrote", path)
