import logging
from typing import Final

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient

logger = logging.getLogger(__name__)

milvus_client: MilvusClient | None = None

small_embedding_model: Final[str] = (
    "sentence-transformers/all-MiniLM-L6-v2"  # 快速, 80 MB
)
big_embedding_model: Final[str] = "BAAI/bge-large-en-v1.5"  # 高精度, 1.3 GB


def get_embedding_dimension(embedding_model: str) -> int:
    # 根据 Embedding 模型选择: all-MiniLM-L6-v2=384, BGE-large=1024
    return 384 if embedding_model == small_embedding_model else 1024


def get_milvus_client(uri: str = "./milvus_demo.db") -> MilvusClient:
    global milvus_client
    if milvus_client is None:
        milvus_client = MilvusClient(uri=uri)
    return milvus_client


def create_milvus_collection(
    coll_name: str = "documents",
    embedding_model: str = small_embedding_model,
):
    client = get_milvus_client()

    if client.has_collection(coll_name):
        logger.info("collection %s already exist, delete old one ...", coll_name)
        client.drop_collection(coll_name)

    dimension = get_embedding_dimension(embedding_model)
    client.create_collection(
        collection_name=coll_name,
        dimension=dimension,  # 向量维度
        metric_type="L2",  # 距离度量: L2 (欧氏距离) 或 IP (内积)
        auto_id=True,
    )
    logger.info("collection created: %s", coll_name)


def create_milvus_collection_with_index(coll_name: str = "documents"):
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
    status = collection.create_index(field_name="vector", index_params=index_params)
    logger.info("index created: %s", status)


if __name__ == "__main__":
    pass
