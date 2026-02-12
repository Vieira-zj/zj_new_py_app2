import logging

import yaml
from pymilvus import MilvusClient


def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("rag_system.log"), logging.StreamHandler()],
    )


def init_config():
    with open("./config.yaml", mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

        if isinstance(config, dict):
            milvus_uri = config["milvus"]["uri"]
            collection_name = config["milvus"]["collection_name"]
            print(
                f"load config: milvus_uri={milvus_uri}, collection_name={collection_name}"
            )


def test_milvus_client():
    _ = MilvusClient(uri="./milvus_test.db")
    print("connected to milvus lite")


if __name__ == "__main__":
    # init_logger()
    init_config()

    # test_milvus_client()
