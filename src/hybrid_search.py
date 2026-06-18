"""混合检索模块：向量检索 + BM25 检索 + RRF 融合排序。"""
import numpy as np

from src.embedding import EmbeddingModel
from src.bm25_index import BM25Index
from src.vector_store import FAISSVectorStore


class HybridSearch:
    """组合向量检索和 BM25 检索，使用 RRF 算法融合排序。"""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        bm25_index: BM25Index,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def search(
        self,
        query: str,
        embedding_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[int]:
        """执行混合检索。

        Args:
            query: 英文查询文本
            embedding_top_k: 向量检索返回数量
            bm25_top_k: BM25检索返回数量
            final_top_k: RRF融合后最终返回数量
            rrf_k: RRF 算法中的 k 参数

        Returns:
            最终排名的文档索引列表
        """
        # 1. 向量检索
        query_vec = self.embedding_model.embed_query(query)
        query_vec = query_vec / np.linalg.norm(query_vec)  # L2 归一化用于余弦相似度
        vec_scores, vec_indices = self.vector_store.search(
            np.array(query_vec), top_k=embedding_top_k
        )

        # 2. BM25 检索
        bm25_indices, bm25_scores = self.bm25_index.search(query, top_k=bm25_top_k)

        # 3. RRF 融合
        rrf_scores = {}

        # 向量检索结果
        for rank, idx in enumerate(vec_indices):
            if idx < 0:
                continue
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)

        # BM25 检索结果
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)

        # 按 RRF 分数降序排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_indices = [idx for idx, _ in sorted_items[:final_top_k]]

        return final_indices
