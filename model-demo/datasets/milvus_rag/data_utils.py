from typing import Final, List

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient
from sentence_transformers import SentenceTransformer

small_embedding_model: Final[str] = (
    "sentence-transformers/all-MiniLM-L6-v2"  # 快速, 80 MB
)
big_embedding_model: Final[str] = "BAAI/bge-large-en-v1.5"  # 高精度, 1.3 GB


def get_embedding_dimension(embedding_model: str) -> int:
    # 根据 Embedding 模型选择: all-MiniLM-L6-v2=384, BGE-large=1024
    return 384 if embedding_model == small_embedding_model else 1024


def get_milvus_client(uri: str = "./milvus_demo.db") -> MilvusClient:
    client = MilvusClient(uri=uri)
    return client


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
    print("embeddings shape:", embeddings.shape)
    return embeddings


# Milvus Utils


def create_milvus_collection(
    coll_name: str = "documents",
    embedding_model: str = small_embedding_model,
):
    client = get_milvus_client()

    if client.has_collection(coll_name):
        print(f"collection {coll_name} already exist, delete old one ...")
        client.drop_collection(coll_name)

    dimension = get_embedding_dimension(embedding_model)
    client.create_collection(
        collection_name=coll_name,
        dimension=dimension,  # 向量维度
        metric_type="L2",  # 距离度量: L2 (欧氏距离) 或 IP (内积)
        auto_id=True,
    )
    print(f"milvus collection {coll_name} created")


async def create_milvus_collection_with_index(coll_name: str = "documents"):
    # schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    ]

    schema = CollectionSchema(fields=fields, description="Documents collection")
    collection = Collection(name=coll_name, schema=schema)

    # 创建 HNSW 索引
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {
            "M": 16,  # 每个节点的连接数
            "efConstruction": 200,  # 构建时的搜索范围
        },
    }
    await collection.create_index(field_name="vector", index_params=index_params)


def insert_milvus_collection(chunks: List[str], coll_name: str):
    embeddings = embed_chunks(chunks)

    data = []
    for _, (emb, chunk) in enumerate(zip(embeddings, chunks)):
        data.append(
            {
                "text": chunk,
                "vector": emb.tolist(),
            }
        )

    client = get_milvus_client()
    client.insert(collection_name=coll_name, data=data)
    print(f"✓ 成功插入 {len(data)} 个文档")


# Main


def test_prepare_data():
    documents = [
        "Milvus is an open-source vector database...",
        "RAG combines information retrieval with language models...",
    ]

    collection_name = "documents"
    create_milvus_collection(collection_name)
    insert_milvus_collection(documents, collection_name)
    print(f"✓ 数据准备完成, 共 {len(documents)} 个文档")


if __name__ == "__main__":
    pass
