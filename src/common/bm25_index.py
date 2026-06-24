"""BM25 标量检索模块。"""
import pickle
from rank_bm25 import BM25Okapi


class BM25Index:
    """基于 rank-bm25 的 BM25 索引，支持英文分词和持久化。"""

    def __init__(self):
        self.corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, texts: list[str]):
        """从文本列表构建 BM25 索引。

        Args:
            texts: 文本列表
        """
        self.corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 10) -> tuple[list[int], list[float]]:
        """检索与查询最相关的文档。

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            (indices, scores): 文档索引列表和对应分数
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # 按分数降序取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        top_scores = [scores[i] for i in top_indices]

        return top_indices, top_scores

    def save(self, path: str):
        """保存索引到磁盘。"""
        with open(path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "bm25": self.bm25}, f)

    def load(self, path: str):
        """从磁盘加载索引。"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.bm25 = data["bm25"]
