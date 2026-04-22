from typing import Any, Final

import torch
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

collection_name: Final[str] = "collection_hello"

milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient:
    global milvus_client
    if milvus_client is None:
        milvus_client = MilvusClient(uri="/tmp/test/milvus_test.db")
        print("connected to milvus lite")
    return milvus_client


def create_milvus_collection():
    client = get_milvus_client()

    schema = client.create_schema()
    schema.add_field(
        field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=False
    )
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=384)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=10240)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding", index_type="IVF_FLAT", metric_type="L2"
    )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    print(f"collection created: {collection_name}")


def drop_milvus_collection():
    client = get_milvus_client()
    client.drop_collection(collection_name=collection_name)
    print(f"collection dropped: {collection_name}")


def check_milvus_collection():
    client = get_milvus_client()

    has = client.has_collection(collection_name)
    print("has collection:", has)

    if has:
        stats = client.get_collection_stats(collection_name)
        print("count:", stats.get("row_count", 0))

        info = client.describe_collection(collection_name)
        print("collection info:", info)


def insert_rand_vector():
    num_vectors = 10
    dim = 384

    t = torch.rand([num_vectors, dim])
    print("data shape:", t.shape)

    vectors = t.tolist()
    records: list[dict[str, Any]] = []
    for rid, vec in enumerate(vectors):
        records.append(
            {
                "id": rid,
                "embedding": vec,
                "content": f"rand text at {rid}",
            }
        )

    client = get_milvus_client()
    client.insert(collection_name, data=records)


def insert_embedding_vector():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    docs = ["猫是一种动物", "狗是人类的朋友", "今天天气不错"]
    doc_embeddings = model.encode(docs)
    print("doc embeddings shape:", doc_embeddings.shape)

    records = [
        {"id": get_new_row_id() + rid, "embedding": embedding, "content": docs[rid]}
        for rid, embedding in enumerate(doc_embeddings.tolist())
    ]

    client = get_milvus_client()
    client.insert(collection_name, data=records)
    print(f"inserted total {len(records)}")


def get_new_row_id() -> int:
    client = get_milvus_client()
    stats = client.get_collection_stats(collection_name)
    return stats.get("row_count", 0)


def test_query_by_ids():
    client = get_milvus_client()
    if not client.has_collection(collection_name):
        raise ValueError(f"collection not exist: {collection_name}")

    results = client.query(
        collection_name=collection_name,
        ids=[10, 11],
        output_fields=["content"],
    )
    for hit in results:
        print("hit content:", hit["content"])


def test_dense_query():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "宠物"
    query_embedding = model.encode(query)
    print("query embedding shape:", query_embedding.shape)

    client = get_milvus_client()
    if not client.has_collection(collection_name):
        raise ValueError(f"collection not exist: {collection_name}")

    results = client.search(
        collection_name=collection_name,
        data=[query_embedding.tolist()],
        anns_field="embedding",
        search_params={
            "metric_type": "L2",  # IP / COSINE
            "params": {},
        },
        limit=2,
        output_fields=["id", "content"],
    )

    print(f"\nquery [{query}]:")
    for hits in results:
        for hit in hits:
            print(f'id={hit["id"]}, content={hit["content"]}')


def delete_vector_by_ids():
    client = get_milvus_client()
    client.delete(
        collection_name=collection_name,
        ids=[0, 1, 2],
    )


if __name__ == "__main__":

    # create_milvus_collection()
    # drop_milvus_collection()

    # insert_rand_vector()
    # insert_embedding_vector()

    check_milvus_collection()

    # test_query_by_ids()
    test_dense_query()

    # delete_vector_by_ids()
