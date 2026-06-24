"""DualIndex：FAISS 向量索引 + BM25 词法索引 + RRF 融合的复合体。

供 chunks 和 abstracts 两类语料共用。
"""
import os
import numpy as np

from src.common.embedding import EmbeddingModel
from src.common.vector_store import FAISSVectorStore
from src.common.bm25_index import BM25Index
from src.rag.hybrid_search import rrf_fuse


class DualIndex:
    """对一份语料同时维护 FAISS 与 BM25 索引，支持混合检索。"""

    def __init__(self, embedding_model: EmbeddingModel, dimension: int):
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.vector_store = FAISSVectorStore(dimension=dimension)
        self.bm25_index = BM25Index()
        self.size = 0

    def build(self, texts: list[str]):
        """从文本列表构建两个索引（向量已 L2 归一化）。"""
        self.vector_store = FAISSVectorStore(dimension=self.dimension)
        self.bm25_index = BM25Index()
        self.size = len(texts)
        if not texts:
            return

        vectors = np.array(self.embedding_model.embed(texts))
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vector_store.add_vectors(vectors)
        self.bm25_index.build(texts)

    def save(self, faiss_dir: str, bm25_path: str):
        os.makedirs(os.path.dirname(bm25_path) or ".", exist_ok=True)
        self.vector_store.save(faiss_dir)
        self.bm25_index.save(bm25_path)

    def load(self, faiss_dir: str, bm25_path: str, size: int):
        self.vector_store = FAISSVectorStore(dimension=self.dimension)
        self.vector_store.load(faiss_dir)
        self.bm25_index = BM25Index()
        self.bm25_index.load(bm25_path)
        self.size = size

    def search(
        self,
        query: str,
        embedding_top_k: int = 10,
        bm25_top_k: int = 10,
        final_top_k: int = 5,
    ) -> list[int]:
        """对单条查询做向量+BM25 混合检索，RRF 融合排序后返回索引。"""
        if self.size == 0:
            return []

        query_vec = np.array(self.embedding_model.embed_query(query))
        query_vec = query_vec / np.linalg.norm(query_vec)
        _, vec_indices = self.vector_store.search(query_vec, top_k=embedding_top_k)
        bm25_indices, _ = self.bm25_index.search(query, top_k=bm25_top_k)

        return rrf_fuse(
            [list(vec_indices), list(bm25_indices)],
            final_top_k=final_top_k,
        )
