import hashlib
import logging
import time
from typing import Any, Dict, List, NotRequired, Optional, TypedDict

import data_utils
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Metrics(TypedDict):
    total_queries: int
    total_retrieval_time: float
    total_rerank_time: float
    total_query_time: float
    total_errors: NotRequired[int]


class RagSystem:

    def __init__(
        self,
        milvus_uri: str = "./milvus_demo.db",
        collection_name: str = "documents",
        embedding_model: str = data_utils.small_embedding_model,
        reranker_model: Optional[str] = "BAAI/bge-reranker-base",
    ):
        self.logger = logging.getLogger(__name__)
        self.cache: dict[str, list] = {}
        self.metrics: Metrics = {
            "total_queries": 0,
            "total_retrieval_time": 0,
            "total_rerank_time": 0,
            "total_query_time": 0,
        }

        # 连接 Milvus
        self.client = data_utils.get_milvus_client(milvus_uri)
        self.collection_name = collection_name

        # 初始化 Embedding 模型
        self.encoder = SentenceTransformer(embedding_model)

        # 初始化重排模型 (可选)
        self.reranker = None
        self.reranker_tokenizer = None
        if reranker_model:
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                reranker_model
            )
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
            self.reranker.eval()
            self.logger.info("✓ 重排模型加载成功")

    def _get_cache_key(self, query: str, top_k: int) -> str:
        key_data = f"{query}_{top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def query(
        self, user_query: str, retrieve_top_k: int = 100, rerank_top_k: int = 10
    ) -> Dict[str, Any]:
        """查询流程: 检索 -> 重排 -> 位置优化 -> 生成"""
        self.metrics["total_queries"] += 1
        query_start = time.time()

        # 1. 向量检索 (粗排)
        self.logger.info("步骤 1: 向量检索 (Top-%d)...", retrieve_top_k)
        retrieval_start = time.time()
        retrieved_docs = self.retrieve(user_query, top_k=retrieve_top_k)
        self.metrics["total_retrieval_time"] += time.time() - retrieval_start
        self.logger.info("✓ 检索到 %d 个文档", len(retrieved_docs))

        if len(retrieved_docs) == 0:
            return {
                "query": user_query,
                "context": "",
                "answer": "no documents retrieved",
            }

        # 2. 重排 (精排)
        self.logger.info("\n步骤 2: 重排 (Top-%d)...", rerank_top_k)
        rerank_start = time.time()
        doc_texts = [doc["text"] for doc in retrieved_docs]
        self.metrics["total_rerank_time"] += time.time() - rerank_start
        reranked_texts = self.rerank(user_query, doc_texts, top_k=rerank_top_k)

        # 更新文档列表 (只保留重排后的文档)
        reranked_docs = [doc for doc in retrieved_docs if doc["text"] in reranked_texts]
        # 按重排顺序重新排序
        text_to_doc: dict = {doc["text"]: doc for doc in reranked_docs}
        reranked_docs: list = [text_to_doc[text] for text in reranked_texts]
        self.logger.info("✓ 重排完成, 返回 Top-%d 个文档", len(reranked_docs))

        # 3. 位置优化构建上下文
        self.logger.info("\n步骤 3: 位置优化构建上下文...")
        context = self.build_context(user_query, reranked_docs)
        self.logger.info("✓ 上下文构建完成 (长度: %d 字符)", len(context))

        # 4. 生成答案 (这里只返回上下文，实际应用中会调用 LLM)
        self.metrics["total_query_time"] += time.time() - query_start
        self.logger.info("\n步骤 4: 上下文已准备就绪\n")

        return {
            "query": user_query,
            "retrieved_docs": retrieved_docs[:5],  # 只返回前 5 个用于展示
            "reranked_docs": reranked_docs,
            "context": context,
            "answer": "[提示] 未配置 LLM, 仅返回上下文.\n\n" + context,
        }

    def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        cache_key = self._get_cache_key(query, top_k)
        if cache_key in self.cache:
            self.logger.info("✓ 使用缓存结果")
            return self.cache[cache_key]

        # 1. 编码查询
        query_vector = self.encoder.encode(query, normalize_embeddings=True)
        query_vector = query_vector.tolist()

        # 2. 在 Milvus 中搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            search_params={"metric_type": "L2", "params": {}},
            output_fields=["text", "doc_id", "title"],
        )

        # 3. 格式化结果
        retrieved_docs: List[dict] = []
        for hit in results[0]:
            entity: Dict[str, str] = hit.get("entity", {})
            retrieved_docs.append(
                {
                    "id": hit.get("id", ""),
                    "text": entity.get("text", ""),
                    "doc_id": entity.get("doc_id", ""),
                    "title": entity.get("title", ""),
                    "score": hit.get("distance", 0.0),
                }
            )

        self.cache[cache_key] = retrieved_docs
        return retrieved_docs

    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[str]:
        """
        重排: 使用 BGE-Reranker 对检索结果进行精排

        1. 重排模型考虑查询和文档的交互, 比单纯向量相似度更准确
        2. 以调整重排后的 Top-K, 平衡准确性和成本
        """
        if (
            self.reranker is None
            or self.reranker_tokenizer is None
            or len(documents) == 0
        ):
            self.logger.info("\n步骤 2: 跳过重排 (未配置 Reranker)...")
            reranked_docs = documents[:top_k]
            self.logger.info("✓ 使用检索结果, 返回 Top-%d 个文档", len(reranked_docs))
            return reranked_docs

        # 构建 "查询-文档" 对
        pairs = [[query, doc] for doc in documents]

        # Tokenize
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            # 计算相关性分数
            scores: Tensor = self.reranker(**inputs).logits.squeeze(-1)
            # 按分数排序
            ranked_indices = scores.argsort(descending=True)
            # 返回 Top-K
            reranked_docs = [documents[idx] for idx in ranked_indices[:top_k]]

        return reranked_docs

    def build_context(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        构建上下文: 位置优化, 突破U型陷阱的关键步骤

        策略:
        - 最相关文档 -> 开头 (利用 Primacy Bias)
        - 次相关文档 -> 中间 (低优先级)
        - 用户问题 -> 结尾 (利用 Recency Bias)
        """
        if len(retrieved_docs) == 0:
            return f"系统提示: 请回答问题. \n\n用户问题: {query}"

        # 按相关性排序 (最相关的在前)
        sorted_docs = sorted(
            retrieved_docs,
            key=lambda x: (
                x.get("score", 0) if isinstance(x.get("score"), (int, float)) else 0
            ),
            reverse=True,  # 分数越高越好 (如果是相似度分数)
        )

        # 构建上下文
        context_parts = []
        # 系统提示
        context_parts.append("系统提示: 请基于以下文档回答问题. \n")

        # 最相关文档 (开头位置 - 利用 Primacy Bias)
        context_parts.append("# 最相关文档 (开头位置)\n")
        top_docs = sorted_docs[: min(3, len(sorted_docs))]
        for i, doc in enumerate(top_docs, 1):
            text = doc.get("text", "")
            context_parts.append(f"文档 {i}: {text}\n")

        # 次相关文档 (中间位置)
        if len(sorted_docs) > 3:
            context_parts.append("\n# 次相关文档 (中间位置)\n")
            secondary_docs = sorted_docs[3 : min(7, len(sorted_docs))]
            for i, doc in enumerate(secondary_docs, 4):
                text = doc.get("text", "")
                context_parts.append(f"文档 {i}: {text}\n")

        # 用户问题 (结尾位置 - 利用 Recency Bias)
        context_parts.append("\n# 用户问题 (结尾位置)\n")
        context_parts.append(f"用户问题: {query}")

        return "\n".join(context_parts)

    def batch_retrieve(self, queries: List[str], batch_size: int = 32):
        # 批量编码
        query_vectors = self.encoder.encode(queries, normalize_embeddings=True)

        all_results = []
        # 批量搜索
        for i in range(0, len(query_vectors), batch_size):
            batch_queries = queries[i : i + batch_size]  # pylint: disable=W0612
            batch_vectors = query_vectors[i : i + batch_size]

            results = self.client.search(
                collection_name=self.collection_name,
                data=batch_vectors.tolist(),
                limit=10,
                search_params={"metric_type": "L2", "params": {}},
                output_fields=["text"],
            )

            all_results.extend(results)

        return all_results

    def get_metrics(self):
        total = self.metrics["total_queries"]
        if total == 0:
            return {}

        return {
            "total_queries": total,
            "avg_query_time": self.metrics["total_query_time"] / total,
            "avg_retrieval_time": self.metrics["total_retrieval_time"] / total,
            "avg_rerank_time": self.metrics["total_rerank_time"] / total,
        }


def test_rag_system():
    rag = RagSystem()
    result = rag.query("What is Milvus?")
    print(
        f"rag results:\n{result["context"]}",
    )


if __name__ == "__main__":
    pass
