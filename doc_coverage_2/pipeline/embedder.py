from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_CANDIDATES = ["BAAI/bge-large-en-v1.5", "all-MiniLM-L6-v2"]
_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
_MODEL = None
_MODEL_NAME = None


def _safe_name(value: str) -> str:
    return value.replace("/", "_")


def _get_model(preferred_model: str | None = None) -> tuple[SentenceTransformer, str]:
    global _MODEL, _MODEL_NAME
    candidates = [preferred_model] if preferred_model else []
    candidates += [name for name in MODEL_CANDIDATES if name != preferred_model]
    if _MODEL is not None and _MODEL_NAME in candidates:
        return _MODEL, _MODEL_NAME
    last_error: Exception | None = None
    for name in candidates:
        if not name:
            continue
        try:
            model = SentenceTransformer(name)
            _MODEL = model
            _MODEL_NAME = name
            return model, name
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Unable to load any sentence-transformers model") from last_error


def _cache_file(prefix: str, texts: list[str], model_name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = sha256((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    return _CACHE_DIR / f"{prefix}_{_safe_name(model_name)}_{digest}.npy"


def embed_texts(texts: list[str], cache_prefix: str, preferred_model: str | None = None) -> tuple[np.ndarray, str]:
    model, model_name = _get_model(preferred_model)
    cache_path = _cache_file(cache_prefix, texts, model_name)
    if cache_path.exists():
        return np.load(cache_path), model_name
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    np.save(cache_path, embeddings)
    return embeddings, model_name
