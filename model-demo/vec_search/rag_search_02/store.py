import logging
from pathlib import Path
from typing import Any, ClassVar

from milvus_lite.server_manager import server_manager_instance
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

logger = logging.getLogger(__name__)


class MilvusStore:

    def __init__(
        self,
        uri: str = "/tmp/test/milvus.db",
        collection: str = "mdsearch_chunks",
        dimension: int | None = 1536,
        description: str = "",
    ) -> None:
        resolved = str(Path(uri).expanduser())
        self._resolved_uri = resolved
        self._client = MilvusClient(uri=resolved)
        self._collection = collection
        self._dimension = dimension
        self._description = description

    def _ensure_collection(self) -> None:
        if self._client.has_collection(self._collection):
            self._check_dimension()
            return

        if self._dimension is None:
            return  # read-only mode: don't create a new collection
        self._create_collection()

    def _create_collection(self) -> None:
        schema = self._client.create_schema(
            enable_dynamic_field=True,
            description=self._description,
        )
        schema.add_field(
            field_name="chunk_hash",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self._dimension
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="heading", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(field_name="heading_level", datatype=DataType.INT64)
        schema.add_field(field_name="start_line", datatype=DataType.INT64)
        schema.add_field(field_name="end_line", datatype=DataType.INT64)
        schema.add_function(
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["sparse_vector"],
            )
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding", index_type="FLAT", metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )

    def _check_dimension(self) -> None:
        if self._dimension is None:
            return  # no dimension specified — skip check (read-only mode)

        try:
            info = self._client.describe_collection(self._collection)
            logger.info("collection info: %s", info)
        except Exception as e:
            logging.warning("describe collection failed: %s", e)

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        result = self._client.upsert(
            collection_name=self._collection,
            data=chunks,
        )
        return (
            result.get("upsert_count", len(chunks))
            if isinstance(result, dict)
            else len(chunks)
        )

    _QUERY_FIELDS: ClassVar[list[str]] = [
        "content",
        "source",
        "heading",
        "chunk_hash",
        "heading_level",
        "start_line",
        "end_line",
    ]

    def search(
        self,
        query_embedding: list[float],
        query_text: str = "",
        top_k: int = 10,
        filter_expr: str = "",
    ) -> list[dict[str, Any]]:
        """Hybrid search: dense vector + BM25 full-text with RRF reranking."""
        stats = self._client.get_collection_stats(self._collection)
        if int(stats.get("row_count", 0)) == 0:
            return []

        req_kwargs: dict[str, Any] = {}
        if filter_expr:
            req_kwargs["expr"] = filter_expr

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            **req_kwargs,
        )

        bm25_req = AnnSearchRequest(
            data=[query_text] if query_text else [""],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=top_k,
            **req_kwargs,
        )

        results = self._client.hybrid_search(
            collection_name=self._collection,
            reqs=[dense_req, bm25_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=self._QUERY_FIELDS,
        )

        if not results or not results[0]:
            return []
        return [{**hit["entity"], "score": hit["distance"]} for hit in results[0]]

    def query(self, filter_expr: str = "") -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "collection_name": self._collection,
            "output_fields": self._QUERY_FIELDS,
            "filter": filter_expr if filter_expr else 'chunk_hash != ""',
        }
        return self._client.query(**kwargs)

    def get_hashes_by_source(self, source: str) -> set[str]:
        """Return all chunk_hash values for a given source file."""
        escaped = escape_filter_value(source)
        results = self._client.query(
            collection_name=self._collection,
            filter=f'source == "{escaped}"',
            output_fields=["chunk_hash"],
        )
        return {r["chunk_hash"] for r in results}

    def get_indexed_sources(self) -> set[str]:
        """Return all distinct source values in the collection."""
        results = self._client.query(
            collection_name=self._collection,
            filter='chunk_hash != ""',
            output_fields=["source"],
        )
        return {r["source"] for r in results}

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a given source file."""
        escaped = escape_filter_value(source)
        self._client.delete(
            collection_name=self._collection,
            filter=f'source == "{escaped}"',
        )

    def delete_by_hashes(self, hashes: list[str]) -> None:
        """Delete chunks by their content hashes (primary keys)."""
        if not hashes:
            return
        self._client.delete(
            collection_name=self._collection,
            ids=hashes,
        )

    def count(self) -> int:
        """Return total number of stored chunks."""
        stats = self._client.get_collection_stats(self._collection)
        return stats.get("row_count", 0)

    def close(self) -> None:
        self._client.close()
        try:
            server_manager_instance.release_server(self._resolved_uri)
        except Exception as e:
            logger.warning("release milvus server failed: %s", e)


def escape_filter_value(value: str) -> str:
    """Escape backslashes and double quotes for Milvus filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    pass
