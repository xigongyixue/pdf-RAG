"""Retriever：基于摘要索引和正文索引的检索接口。"""
import json
import os

from src.common.dual_index import DualIndex


class Retriever:
    """封装 abstract_index 与 chunk_index，按需加载 chunk JSON。"""

    def __init__(
        self,
        abstract_index: DualIndex,
        chunk_index: DualIndex,
        meta: dict,
        chunks_dir: str,
    ):
        self.abstract_index = abstract_index
        self.chunk_index = chunk_index
        self.meta = meta
        self.chunks_dir = chunks_dir
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
        indices = self.abstract_index.search(
            query,
            embedding_top_k=embedding_top_k,
            bm25_top_k=bm25_top_k,
            final_top_k=top_k,
        )
        articles = self.meta.get("articles", [])
        result = []
        for idx in indices:
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

        Returns:
            list of {article, index, section, content}
        """
        # 限定文章时扩大召回上限再过滤
        overscan = 3 if article_filter else 1
        indices = self.chunk_index.search(
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

        results = []
        seen = set()
        for idx in indices:
            if allowed_ranges is not None and not _in_any_range(idx, allowed_ranges):
                continue
            if idx in seen:
                continue
            seen.add(idx)
            chunk = self.get_chunk(idx)
            if chunk:
                results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    # ─── chunk 按需加载 ───────────────────────────────────
    def get_chunk(self, global_idx: int) -> dict | None:
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
