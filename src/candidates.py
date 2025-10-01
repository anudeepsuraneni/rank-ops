import os
import pickle
from typing import Optional
import numpy as np
from src.train import build_faiss

_FAISS_READY=False
_FAISS_INDEX=None
_ITEM_VECS=None
_ITEM_IDS=None
_COVIS=None

def _load_als(path: str = 'models/als.pkl') -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'ALS model not found at {path}')
    with open(path,'rb') as f:
        return pickle.load(f)

def _ensure_faiss(als_obj: dict, index_path: str = 'models/faiss.index', nlist: int = 100) -> None:
    global _FAISS_READY,_FAISS_INDEX,_ITEM_VECS,_ITEM_IDS
    if _FAISS_READY: return
    import faiss
    model = als_obj['model']; items = als_obj['items']
    item_vecs = np.ascontiguousarray(model.item_factors.astype(np.float32))
    _ITEM_VECS = item_vecs; _ITEM_IDS = list(map(int, items))
    if not os.path.exists(index_path):
        build_faiss(index_path)
    _FAISS_INDEX=faiss.read_index(index_path); _FAISS_READY=True

def _ensure_covis(path: str='models/covis.pkl')->None:
    global _COVIS
    if _COVIS is not None: return
    if os.path.exists(path):
        with open(path,'rb') as f: _COVIS = pickle.load(f)
    else:
        _COVIS = {}

def candidates_als(user_id:int, k:int, als_obj:Optional[dict]=None)->list[int]:
    if als_obj is None: als_obj=_load_als()
    model = als_obj['model']; users=als_obj['users']; items=als_obj['items']
    if user_id not in users: return []
    uid = users.index(user_id)
    from scipy.sparse import csr_matrix
    recs = model.recommend(uid, csr_matrix((1,len(items))), N=k)
    return list(map(int, recs[0]))

def candidates_faiss(user_id:int, k:int, als_obj:Optional[dict]=None)->list[int]:
    if als_obj is None: als_obj=_load_als()
    _ensure_faiss(als_obj)
    model = als_obj['model']; users=als_obj['users']
    if user_id not in users: return []
    uid = users.index(user_id)
    uvec = model.user_factors[uid].astype(np.float32).reshape(1,-1)
    assert _FAISS_INDEX is not None, "FAISS index is not initialized"
    index = _FAISS_INDEX
    k = min(k, index.ntotal)
    D = np.empty((1, k), dtype=np.float32)
    I = np.empty((1, k), dtype=np.int64)
    index.search(uvec.shape[0], uvec, k, D, I)
    return [int(_ITEM_IDS[i]) for i in I[0] if _ITEM_IDS and i>=0]

def candidates_covis(user_id:int, k:int, recent_items:Optional[list[int]]=None)->list[int]:
    _ensure_covis()
    seeds=(recent_items or [])[:20]
    from collections import Counter
    c=Counter()
    for it in seeds:
        for j in (_COVIS.get(int(it),[]) if _COVIS else []): c[j]+=1
    return [int(x) for x,_ in c.most_common(k)]

def get_candidates(user_id:int,k:int,backend:str='ALS',als_obj:Optional[dict]=None,recent_items:Optional[list[int]]=None)->list[int]:
    if backend=='ALS': return candidates_als(user_id,k,als_obj=als_obj)
    if backend=='FAISS': return candidates_faiss(user_id,k,als_obj=als_obj)
    if backend=='COVIS': return candidates_covis(user_id,k,recent_items=recent_items)
    raise ValueError(f'Unknown backend {backend}')
