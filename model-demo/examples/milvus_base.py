from typing import Final

from pymilvus import MilvusClient

collection: Final[str] = "collection_hello"


def get_milvus_client() -> MilvusClient:
    client = MilvusClient(uri="/tmp/test/milvus_test.db")
    print("connected to milvus lite")
    return client


def create_milvus_collection():
    pass


def check_milvus_collection():
    client = get_milvus_client()

    has = client.has_collection(collection)
    print("has collection:", has)

    if has:
        stats = client.get_collection_stats(collection)
        print("count:", stats.get("row_count", 0))

        info = client.describe_collection(collection)
        print("collection info:", info)


if __name__ == "__main__":

    create_milvus_collection()
    # check_milvus_collection()
