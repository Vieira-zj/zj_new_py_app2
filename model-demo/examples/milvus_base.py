from typing import Any, Final

import torch
from pymilvus import DataType, MilvusClient

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
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=128)

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


def insert_vector():
    num_vectors = 10
    dim = 128

    ids = list(range(num_vectors))
    t = torch.rand([num_vectors, dim])
    print("data shape:", t.shape)

    vectors = t.tolist()
    records: list[dict[str, Any]] = []
    for rid, vec in zip(ids, vectors):
        records.append(
            {
                "id": rid,
                "embedding": vec,
            }
        )

    client = get_milvus_client()
    client.insert(collection_name, data=records)


def delete_vector():
    client = get_milvus_client()
    client.delete(collection_name=collection_name, ids=[0, 1, 2])


if __name__ == "__main__":

    # create_milvus_collection()
    # drop_milvus_collection()

    # insert_vector()
    # delete_vector()

    check_milvus_collection()
