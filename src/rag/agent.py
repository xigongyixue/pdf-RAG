"""Agentic RAG：根据问题类型分发到不同的检索策略。"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from src.rag.retriever import Retriever


_STRATEGY_PROMPT = """You are a RAG query planner. Given a user's question (Chinese or English), output a JSON plan with these fields:

- "strategy": one of "ABSTRACT_FIRST", "DIRECT_CHUNK", "MULTI_HOP"
- "english_query": the question translated/normalized into an English retrieval query (single string)
- "sub_queries": only for MULTI_HOP, a list of 2-4 English sub-queries (otherwise []).

Strategy guide:
- ABSTRACT_FIRST: the question targets a specific topic/method/dataset (e.g. "Transformer 在 ImageNet 中的应用"). Locate relevant articles via abstracts first.
- DIRECT_CHUNK: the question asks for a cross-article enumeration of attributes/features (e.g. "哪些数据库使用了 LRU 缓存"). Search chunks directly.
- MULTI_HOP: the question compares/contrasts multiple entities or is composite (e.g. "BERT 和 GPT-3 的区别"). Decompose into sub-queries, each suitable for ABSTRACT_FIRST.

Output ONLY the JSON object, no explanation."""


_ANSWER_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the provided "
    "context extracted from PDF documents in English. Each fragment is labeled "
    "[N] <ARTICLE> SECTION. You MUST answer in Chinese (中文). If the context does "
    "not contain enough information, say '根据提供的文档无法回答该问题'."
)


class AgenticRAG:
    """Query planner + 三种检索策略的执行体。"""

    def __init__(
        self,
        retriever: Retriever,
        llm_client,
        llm_model: str,
        retrieval_cfg: dict,
    ):
        self.retriever = retriever
        self.client = llm_client
        self.model = llm_model
        self.cfg = retrieval_cfg

    # ─── 入口 ──────────────────────────────────────────────
    def answer(self, question: str) -> tuple[str, list[dict]]:
        plan = self._classify_and_rewrite(question)
        strategy = plan.get("strategy", "ABSTRACT_FIRST")
        english_query = plan.get("english_query") or question
        sub_queries = plan.get("sub_queries") or []
        print(f"[Agent] strategy={strategy}, english_query={english_query!r}", end="")
        if strategy == "MULTI_HOP":
            print(f", sub_queries={sub_queries}")
        else:
            print()

        if strategy == "DIRECT_CHUNK":
            sources = self._direct_chunk(english_query)
        elif strategy == "MULTI_HOP" and sub_queries:
            sources = self._multi_hop(sub_queries)
        else:
            sources = self._abstract_first(english_query)

        if not sources:
            return "未找到相关文档内容。", []
        return self._compose_answer(question, sources), sources

    # ─── 策略 1: 摘要优先 ─────────────────────────────────
    def _abstract_first(self, english_query: str) -> list[dict]:
        article_top_k = self.cfg.get("article_top_k", 3)
        articles = self.retriever.search_articles(english_query, top_k=article_top_k)
        print(f"[Agent] abstract_first -> articles: {articles}")
        if not articles:
            return self._direct_chunk(english_query)

        return self.retriever.search_chunks(
            english_query,
            top_k=self.cfg["final_top_k"],
            embedding_top_k=self.cfg["embedding_top_k"],
            bm25_top_k=self.cfg["bm25_top_k"],
            article_filter=articles,
        )

    # ─── 策略 2: 直接检索 chunk ──────────────────────────
    def _direct_chunk(self, english_query: str) -> list[dict]:
        return self.retriever.search_chunks(
            english_query,
            top_k=self.cfg["final_top_k"],
            embedding_top_k=self.cfg["embedding_top_k"],
            bm25_top_k=self.cfg["bm25_top_k"],
        )

    # ─── 策略 3: 多跳并行 ────────────────────────────────
    def _multi_hop(self, sub_queries: list[str]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(4, len(sub_queries))) as ex:
            results = list(ex.map(self._abstract_first, sub_queries))

        merged: list[dict] = []
        seen = set()
        for group in results:
            for c in group:
                key = (c["article"], c["index"])
                if key in seen:
                    continue
                seen.add(key)
                merged.append(c)
        return merged

    # ─── LLM: 分类+改写 ───────────────────────────────────
    def _classify_and_rewrite(self, question: str) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _STRATEGY_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_json(raw)
        except Exception as e:
            print(f"[Agent] classify 失败，降级 ABSTRACT_FIRST: {e}")
            return {"strategy": "ABSTRACT_FIRST", "english_query": question, "sub_queries": []}

    # ─── LLM: 生成答案 ────────────────────────────────────
    def _compose_answer(self, question: str, sources: list[dict]) -> str:
        context_parts = [
            f"[{s['index']}] <{s['article']}> {s['section']}:\n{s['content']}"
            for s in sources
        ]
        context = "\n\n---\n\n".join(context_parts)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _ANSWER_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(raw: str) -> dict:
    """从 LLM 输出中宽松解析 JSON（去 fenced code，再 fallback 到大括号截取）。"""
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise
