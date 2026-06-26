"""Agentic RAG：将查询分为单问题和多问题两类，并执行摘要改写+直接检索融合。"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from src.rag.retriever import Retriever


_STRATEGY_PROMPT = """You are a RAG query planner. Given a user's question (Chinese or English), output a JSON plan with these fields:

- "strategy": one of "SINGLE_QUERY", "MULTI_QUERY"
- "english_query": the question translated/normalized into an English retrieval query (single string)
- "sub_queries": only for MULTI_QUERY, a list of 2-4 English sub-queries (otherwise []).

Strategy guide:
- SINGLE_QUERY: the question can be answered as one focused query.
- MULTI_QUERY: the question compares multiple entities, contains multiple independent questions, or needs decomposition.

Output ONLY the JSON object, no explanation."""


_REWRITE_PROMPT = """You rewrite a user's question into a better English retrieval query using retrieved article abstracts.

Requirements:
- Keep the original intent.
- Add important paper-specific terms found in the abstracts when helpful.
- Output ONLY one English retrieval query, no explanation."""


_FOLLOWUP_PROMPT = """You turn a follow-up question into a standalone question for retrieval.

Use the previous question, previous answer, and previous cited sources only to resolve references like "it", "this method", "the paper", "继续", "它".
If the new question is already standalone, keep its meaning unchanged.
Output ONLY the standalone question in English, no explanation."""


_ANSWER_PROMPT = (
    "You are a helpful assistant. Answer the user's question based on the provided "
    "context extracted from PDF documents in English. Each fragment is labeled "
    "[N] <ARTICLE> SECTION. You MUST answer in Chinese (中文). If the context does "
    "not contain enough information, say '根据提供的文档无法回答该问题'."
)


class AgenticRAG:
    """Query planner + 单问题/多问题检索执行体。"""

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
        self.last_turn: dict | None = None

    def reset_context(self):
        """清空上一轮问答上下文。"""
        self.last_turn = None

    # ─── 入口 ───────────────────────────────────
    def answer(self, question: str) -> tuple[str, list[dict]]:
        standalone_question = self._rewrite_followup(question)
        if standalone_question != question:
            print(f"[Agent] followup_query={standalone_question!r}")

        plan = self._classify_and_rewrite(standalone_question)
        strategy = plan.get("strategy", "SINGLE_QUERY")
        english_query = plan.get("english_query") or question
        sub_queries = plan.get("sub_queries") or []

        if strategy not in {"SINGLE_QUERY", "MULTI_QUERY"}:
            strategy = "MULTI_QUERY" if sub_queries else "SINGLE_QUERY"

        print(f"[Agent] strategy={strategy}, english_query={english_query!r}", end="")
        if strategy == "MULTI_QUERY":
            print(f", sub_queries={sub_queries}")
        else:
            print()

        if strategy == "MULTI_QUERY" and sub_queries:
            sources = self._multi_query(sub_queries)
        else:
            sources = self._single_query(english_query)

        if not sources:
            answer = "未找到相关文档内容。"
            self._remember_turn(question, standalone_question, answer, [])
            return answer, []

        answer = self._compose_answer(question, standalone_question, sources)
        self._remember_turn(question, standalone_question, answer, sources)
        return answer, sources

    # ─── 单问题查询: 摘要检索 → 摘要改写检索 + 直接检索 → 合并 ───
    def _single_query(self, english_query: str) -> list[dict]:
        article_top_k = self.cfg.get("article_top_k", 3)
        summaries = self.retriever.search_article_summaries(
            english_query,
            top_k=article_top_k,
            embedding_top_k=self.cfg["embedding_top_k"],
            bm25_top_k=self.cfg["bm25_top_k"],
        )
        articles = [s["article"] for s in summaries]
        print(f"[Agent] single_query -> articles: {articles}")

        rewritten_query = self._rewrite_with_abstracts(english_query, summaries)
        print(f"[Agent] rewritten_query={rewritten_query!r}")

        rewritten_sources = []
        if articles:
            rewritten_sources = self.retriever.search_chunks(
                rewritten_query,
                top_k=self.cfg["final_top_k"],
                embedding_top_k=self.cfg["embedding_top_k"],
                bm25_top_k=self.cfg["bm25_top_k"],
                article_filter=articles,
            )

        direct_sources = self.retriever.search_chunks(
            english_query,
            top_k=self.cfg["final_top_k"],
            embedding_top_k=self.cfg["embedding_top_k"],
            bm25_top_k=self.cfg["bm25_top_k"],
        )

        return self._merge_sources(rewritten_sources, direct_sources)

    # ─── 多问题查询: 每个子问题走单问题流程，再合并 ────────────
    def _multi_query(self, sub_queries: list[str]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(4, len(sub_queries))) as ex:
            results = list(ex.map(self._single_query, sub_queries))
        return self._merge_sources(*results)

    # ─── LLM: 分类+改写 ──────────────────────────
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
            print(f"[Agent] classify 失败，降级 SINGLE_QUERY: {e}")
            return {"strategy": "SINGLE_QUERY", "english_query": question, "sub_queries": []}

    def _rewrite_with_abstracts(self, english_query: str, summaries: list[dict]) -> str:
        if not summaries:
            return english_query

        abstract_parts = []
        for i, summary in enumerate(summaries, start=1):
            content = (summary.get("content") or "").strip()
            abstract_parts.append(
                f"[{i}] <{summary['article']}> {summary.get('title', '')}\n{content[:1200]}"
            )
        abstracts = "\n\n---\n\n".join(abstract_parts)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _REWRITE_PROMPT},
                    {"role": "user", "content": f"Abstracts:\n{abstracts}\n\nQuestion: {english_query}"},
                ],
                temperature=0.1,
            )
            rewritten = resp.choices[0].message.content.strip().strip('"')
            return rewritten or english_query
        except Exception as e:
            print(f"[Agent] rewrite 失败，使用原查询: {e}")
            return english_query

    def _merge_sources(self, *source_groups: list[dict]) -> list[dict]:
        """合并多路 chunk 结果；重复块保留最高分，并按文章聚合、块顺序输出。"""
        merged: dict[tuple[str, int], dict] = {}
        for group in source_groups:
            for source in group:
                key = (source["article"], source["index"])
                old = merged.get(key)
                if old is None or source.get("score", 0.0) > old.get("score", 0.0):
                    merged[key] = source

        chunks = list(merged.values())
        chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        chunks = chunks[: self.cfg["final_top_k"]]

        by_article: dict[str, list[dict]] = {}
        article_scores: dict[str, float] = {}
        for chunk in chunks:
            article = chunk["article"]
            by_article.setdefault(article, []).append(chunk)
            article_scores[article] = max(article_scores.get(article, 0.0), chunk.get("score", 0.0))

        final_results = []
        for article in sorted(by_article, key=lambda x: article_scores[x], reverse=True):
            final_results.extend(sorted(by_article[article], key=lambda x: x["index"]))
        return final_results

    def _rewrite_followup(self, question: str) -> str:
        """根据上一轮问答将追问改写为独立问题。"""
        if not self.last_turn:
            return question

        source_summary = ", ".join(
            f"{s['article']}#{s['index']}"
            for s in self.last_turn.get("sources", [])[:8]
        )
        previous_answer = (self.last_turn.get("answer") or "")[:1200]
        previous_question = self.last_turn.get("standalone_question") or self.last_turn.get("question") or ""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _FOLLOWUP_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Previous question: {previous_question}\n"
                            f"Previous answer: {previous_answer}\n"
                            f"Previous sources: {source_summary}\n\n"
                            f"New question: {question}"
                        ),
                    },
                ],
                temperature=0.1,
            )
            rewritten = resp.choices[0].message.content.strip().strip('"')
            return rewritten or question
        except Exception as e:
            print(f"[Agent] followup rewrite 失败，使用原问题: {e}")
            return question

    def _remember_turn(
        self,
        question: str,
        standalone_question: str,
        answer: str,
        sources: list[dict],
    ):
        """保存最近一轮问答，用于下一轮追问。"""
        self.last_turn = {
            "question": question,
            "standalone_question": standalone_question,
            "answer": answer,
            "sources": [
                {
                    "article": s.get("article"),
                    "index": s.get("index"),
                    "section": s.get("section"),
                    "score": s.get("score", 0.0),
                }
                for s in sources
            ],
        }

    # ─── LLM: 生成答案 ──────────────────────────────
    def _compose_answer(self, question: str, standalone_question: str, sources: list[dict]) -> str:
        context_parts = [
            f"[{s['index']}] <{s['article']}> {s['section']}:\n{s['content']}"
            for s in sources
        ]
        context = "\n\n---\n\n".join(context_parts)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _ANSWER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Original question: {question}\n"
                        f"Standalone question for retrieval: {standalone_question}"
                    ),
                },
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
