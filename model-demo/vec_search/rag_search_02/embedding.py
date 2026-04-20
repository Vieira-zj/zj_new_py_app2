from typing import Callable, Protocol, runtime_checkable

import torch
from sentence_transformers import SentenceTransformer


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface every embedding backend must satisfy."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedding:
    """sentence-transformers embedding provider."""

    _DEFAULT_BATCH_SIZE = 512

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 0
    ) -> None:
        self._model_name = model_name
        self._st_model = SentenceTransformer(
            model_name, device=_detect_device(), trust_remote_code=True
        )
        self._dimension = self._st_model.get_sentence_embedding_dimension() or 384
        self._batch_size = batch_size if batch_size > 0 else self._DEFAULT_BATCH_SIZE

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return batched_embed(texts, self._embed_batch, self._batch_size)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._st_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


def batched_embed(
    texts: list[str], embed_fn: Callable[[list[str]]], batch_size: int
) -> list[list[float]]:
    """Split *texts* into batches and call *embed_fn* on each."""
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if len(texts) <= batch_size:
        return embed_fn(texts)

    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        results.extend(embed_fn(texts[i : i + batch_size]))
    return results


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    pass
