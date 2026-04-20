import logging
from pathlib import Path
from typing import Any

from .chunker import Chunk, chunk_markdown
from .embedding import EmbeddingProvider, LocalEmbedding
from .reranker import rerank
from .scanner import ScannedFile, scan_paths
from .store import MilvusStore, escape_filter_value

logger = logging.getLogger(__name__)


class RagSearch:
    def __init__(
        self,
        paths: list[str],
        milvus_uri: str = "/tmp/test/milvus.db",
        collection: str = "mdsearch_chunks",
        description: str = "",
        embedding_batch_size: int = 0,
        max_chunk_size: int = 1500,
        reranker_model: str = "",
    ):
        self._paths = paths
        self._max_chunk_size = max_chunk_size
        self._embedder: EmbeddingProvider = LocalEmbedding(
            batch_size=embedding_batch_size
        )
        self._store = MilvusStore(
            uri=milvus_uri,
            collection=collection,
            dimension=self._embedder.dimension,
            description=description,
        )
        self._reranker_model = reranker_model

    def index(self, force: bool = False) -> int:
        """Scan paths and index all markdown files.

        Returns the number of chunks indexed. Also removes chunks for
        files that no longer exist on disk (deleted-file cleanup).
        """
        md_files = scan_paths(self._paths)

        total, failed = 0, 0
        active_sources: set[str] = set()
        for f in md_files:
            active_sources.add(str(f.path))
            try:
                n = self._index_file(f, force=force)
                total += n
            except Exception as e:
                failed += 1
                logger.exception("failed to index %s, skipping: %s", f.path, e)

        # clean up chunks for files that no longer exist
        indexed_sources = self._store.get_indexed_sources()
        for source in indexed_sources:
            if source not in active_sources:
                self._store.delete_by_source(source)
                logger.info("removed stale chunks for deleted file: %s", source)

        if failed:
            logger.warning(
                "indexed %d chunks from %d files (%d files failed)",
                total,
                len(md_files) - failed,
                failed,
            )
        else:
            logger.info("indexed %d chunks from %d files", total, len(md_files))

        return total

    def _index_file(self, f: ScannedFile, force: bool = False) -> int:
        source = str(f.path)
        text = f.path.read_text(encoding="utf-8")
        chunks = chunk_markdown(
            text,
            source=source,
            max_chunk_size=self._max_chunk_size,
        )

        # compute composite chunk IDs (matching OpenClaw format)
        model_name = self._embedder.model_name
        chunk_ids = {c.compute_chunk_id(model_name) for c in chunks}
        old_ids = self._store.get_hashes_by_source(source)

        # delete stale chunks that are no longer in the file
        stale = old_ids - chunk_ids
        if stale:
            self._store.delete_by_hashes(list(stale))

        if not chunks:
            return 0

        if not force:
            # only embed chunks whose ID doesn't already exist
            chunks = [
                c for c in chunks if c.compute_chunk_id(model_name) not in old_ids
            ]
            if not chunks:
                return 0

        return self._embed_and_store(chunks)

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        model_name = self._embedder.model_name
        contents = [c.content for c in chunks]
        embeddings = self._embedder.embed(contents)

        records: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.compute_chunk_id(model_name)
            records.append(
                {
                    "chunk_hash": chunk_id,
                    "embedding": embeddings[i],
                    "content": chunk.content,
                    "source": chunk.source,
                    "heading": chunk.heading,
                    "heading_level": chunk.heading_level,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                }
            )

        return self._store.upsert(records)

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across indexed chunks."""
        filter_expr = ""
        if source_prefix is not None:
            prefix = str(Path(source_prefix).expanduser().resolve())
            escaped = escape_filter_value(prefix)
            filter_expr = f'source like "{escaped}%"'

        embeddings = self._embedder.embed([query])
        fetch_k = top_k * 3 if self._reranker_model else top_k
        results = self._store.search(
            embeddings[0], query_text=query, top_k=fetch_k, filter_expr=filter_expr
        )

        if self._reranker_model and results:
            results = rerank(
                query, results, model_name=self._reranker_model, top_k=top_k
            )
        return results


# Test


def test_rag_index():
    rag = RagSearch(paths=[""])
    rag.index()


def test_rag_search():
    rag = RagSearch(paths=[""])
    rag.search("")


if __name__ == "__main__":
    pass
