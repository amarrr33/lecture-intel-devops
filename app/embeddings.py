from typing import List
import numpy as np

_EMB_MODEL = None
_EMB_MODEL_NAME = None

def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _EMB_MODEL, _EMB_MODEL_NAME
    if _EMB_MODEL is None or _EMB_MODEL_NAME != model_name:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer(model_name)
        _EMB_MODEL_NAME = model_name
    return _EMB_MODEL

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = get_embedder(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)