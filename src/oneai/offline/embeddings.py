"""Shared embedding helper: embed(text) -> list[float]."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _hasher():
    from sklearn.feature_extraction.text import HashingVectorizer

    return HashingVectorizer(n_features=DIM, alternate_sign=False, ngram_range=(1, 2), norm=None)


def backend() -> str:
    return "minilm" if _model() is not None else "hashing"


def embed_many(texts: Sequence[str]) -> np.ndarray:
    model = _model()
    if model is not None:
        vecs = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
    vecs = _hasher().transform(list(texts)).toarray().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def embed(text: str) -> List[float]:
    return embed_many([text])[0].tolist()
