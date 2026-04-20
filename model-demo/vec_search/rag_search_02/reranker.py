import logging
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cap cross-encoder sequence length to avoid OOM on long documents.
_MAX_RERANK_TOKENS = 512

_torch_cache: dict[str, Any] = {}


def rerank(
    query: str, results: list[dict[str, Any]], model_name: str, top_k: int = 0
) -> list[dict[str, Any]]:
    return _rerank_torch(query, results, model_name, top_k)


def _rerank_torch(
    query: str, results: list[dict[str, Any]], model_name: str, top_k: int
) -> list[dict[str, Any]]:
    """Rerank using sentence-transformers CrossEncoder backend."""
    model = _load_torch_model(model_name)

    pairs = [(query, r["content"]) for r in results]
    raw_scores = model.predict(pairs)

    scores = [float(s) for s in raw_scores]
    scored = [{**r, "score": s} for r, s in zip(results, scores, strict=True)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k] if top_k > 0 else scored


def _load_torch_model(model_name: str) -> Any:
    """Load a sentence-transformers CrossEncoder model."""
    if model_name in _torch_cache:
        return _torch_cache[model_name]

    model = CrossEncoder(model_name, max_length=_MAX_RERANK_TOKENS)
    logger.info("loaded PyTorch cross-encoder reranker: %s", model_name)

    if model_name not in _torch_cache:
        _torch_cache[model_name] = model
    return _torch_cache[model_name]


if __name__ == "__main__":
    pass
