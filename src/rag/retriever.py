"""Retriever：基于摘要索引和正文索引的检索接口。"""
import json
import os
from typing import Optional

from src.common.dual_index import DualIndex
from src.common.db_manager import DatabaseManager


class Retriever:
    """封装 abstract_index 与 chunk_index，支持从数据库或文件加载 chunk。"""

    def __init__(
        self,
        abstract_index: DualIndex,
        chunk_index: DualIndex,
        meta: dict,
        chunks_dir: str,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.abstract_index = abstract_index
        self.chunk_index = chunk_index
        self.meta = meta
        self.chunks_dir = chunks_dir
        self.db_manager = db_manager
        self._chunk_cache: dict[str, list[dict]] = {}

    # ─── 文章级检索 ────────────────────────────────────────
    def search_articles(
        self,
        query: str,
        top_k: int = 3,
        embedding_top_k: int = 10,
        bm25_top_k: int = 10,
    ) -> list[str]:
        """根据 query 在 abstract 索引中检索，返回文章名列表。"""
        results = self.abstract_index.search(
            query,
            embedding_top_k=embedding_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=top_k,
        )
        articles = self.meta.get("articles", [])
        result = []
        for idx, _ in results:
            if 0 <= idx < len(articles):
                result.append(articles[idx]["name"])
        return result

    # ─── 正文 chunk 检索 ──────────────────────────────────
    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        embedding_top_k: int = 10,
        bm25_top_k: int = 10,
        article_filter: list[str] | None = None,
    ) -> list[dict]:
        """在正文索引中检索 chunk；可选限定到某些文章。

        返回的 chunk 按以下规则排序：
        1. 先按文章聚合（同一文章的块放在一起）
        2. 同一文章内按原文章节顺序排列

        Returns:
            list of {article, index, section, content, score}
        """
        # 限定文章时扩大召回上限再过滤
        overscan = 3 if article_filter else 1
        results = self.chunk_index.search(
            query,
            embedding_top_k=embedding_top_k * overscan,
            bm25_top_k=bm25_top_k * overscan,
            final_top_k=top_k * overscan,
        )

        allowed_ranges = None
        if article_filter:
            allowed_ranges = []
            for art in self.meta.get("articles", []):
                if art["name"] in article_filter:
                    allowed_ranges.append(tuple(art["chunk_range"]))

        # 收集有效的 (idx, score) 对
        scored_indices = []
        for idx, score in results:
            if allowed_ranges is not None and not _in_any_range(idx, allowed_ranges):
                continue
            scored_indices.append((idx, score))
            if len(scored_indices) >= top_k:
                break

        # 获取每个索引对应的文章和块信息
        chunks_with_info = []
        seen = set()
        for idx, score in scored_indices:
            if idx in seen:
                continue
            seen.add(idx)
            chunk = self.get_chunk(idx)
            if chunk:
                chunk["score"] = score
                chunks_with_info.append(chunk)

        # 按文章聚合，同一文章内按章节顺序（全局索引）排序
        # 文章顺序按首块的相关性分数确定
        by_article: dict[str, list[dict]] = {}
        article_scores: dict[str, float] = {}
        
        for chunk in chunks_with_info:
            article = chunk["article"]
            by_article.setdefault(article, []).append(chunk)
            # 记录文章的最高分数（用于文章排序）
            if article not in article_scores or chunk["score"] > article_scores[article]:
                article_scores[article] = chunk["score"]

        # 同一文章内按全局索引（章节顺序）排序
        for article in by_article:
            by_article[article].sort(key=lambda x: x["index"])

        # 按文章分数排序文章，然后合并结果
        sorted_articles = sorted(by_article.keys(), key=lambda x: article_scores[x], reverse=True)
        
        final_results = []
        for article in sorted_articles:
            final_results.extend(by_article[article])

        return final_results[:top_k]

    # ─── chunk 按需加载 ───────────────────────────────────
    def get_chunk(self, global_idx: int) -> dict | None:
        """优先从数据库加载块，回退到文件加载。"""
        # 1. 优先尝试从数据库加载
        if self.db_manager:
            chunk = self._get_chunk_from_db(global_idx)
            if chunk:
                return chunk
        
        # 2. 回退到文件加载
        return self._get_chunk_from_file(global_idx)
    
    def _get_chunk_from_db(self, global_idx: int) -> dict | None:
        """从数据库加载块。"""
        try:
            chunk_data = self.db_manager.get_chunk(global_idx)
            if chunk_data:
                return {
                    "article": chunk_data['article_name'],
                    "index": chunk_data['global_index'],
                    "section": chunk_data['section'],
                    "content": chunk_data['content'],
                }
        except Exception as e:
            print(f"从数据库加载块失败: {e}")
        return None
    
    def _get_chunk_from_file(self, global_idx: int) -> dict | None:
        """从文件加载块。"""
        for art in self.meta.get("articles", []):
            lo, hi = art["chunk_range"]
            if lo <= global_idx <= hi:
                local_idx = global_idx - lo
                if art["name"] not in self._chunk_cache:
                    fpath = os.path.join(self.chunks_dir, art["chunks_file"])
                    with open(fpath, "r", encoding="utf-8") as f:
                        self._chunk_cache[art["name"]] = json.load(f)
                chunks = self._chunk_cache[art["name"]]
                c = chunks[local_idx]
                return {
                    "article": art["name"],
                    "index": global_idx,
                    "section": c.get("section", "?"),
                    "content": c.get("content", ""),
                }
        return None


def _in_any_range(idx: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= idx <= hi for lo, hi in ranges)
