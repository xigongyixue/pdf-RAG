"""FAISS 向量存储模块：存储和检索向量。"""
import os
import json
import faiss
import numpy as np


class FAISSVectorStore:
    """基于 FAISS 的向量存储，支持保存和加载。"""

    def __init__(self, dimension: int = 2048):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度（余弦相似度需归一化）

    def add_vectors(self, vectors: np.ndarray):
        """向索引中添加向量。向量需已做 L2 归一化。"""
        self.index.add(vectors.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """检索最相似的 top_k 个向量。

        Args:
            query_vector: 查询向量 (1, dim)
            top_k: 返回数量

        Returns:
            (scores, indices): 相似度分数和对应索引
        """
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k)
        return scores[0], indices[0]

    def save(self, dir_path: str):
        """保存索引到磁盘。"""
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))

    def load(self, dir_path: str):
        """从磁盘加载索引。"""
        path = os.path.join(dir_path, "index.faiss")
        if os.path.exists(path):
            self.index = faiss.read_index(path)
        else:
            raise FileNotFoundError(f"FAISS index not found at {path}")
