import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from store import get_milvus_client, small_embedding_model

logger = logging.getLogger(__name__)


def chunk_document(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    start, end = 0, 0
    chunks: List[str] = []

    while end < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重叠, 保证上下文连贯

    return chunks


def embed_chunks(
    chunks: List[str], embedding_model: str = small_embedding_model
) -> np.ndarray:
    encoder = SentenceTransformer(embedding_model)
    embeddings = encoder.encode(chunks, normalize_embeddings=True)
    logger.info("embeddings shape: %s", embeddings.shape)
    return embeddings


def insert_embeddings(chunks: List[str], coll_name: str):
    embeddings = embed_chunks(chunks)

    data = []
    for emb, chunk in zip(embeddings, chunks):
        data.append(
            {
                "text": chunk,
                "vector": emb.tolist(),
            }
        )

    client = get_milvus_client()
    client.insert(collection_name=coll_name, data=data)
    logger.info("✓ 成功插入 %d 个文档", len(data))


if __name__ == "__main__":
    pass
