"""RAG 主流程：双索引（abstract + chunk）生命周期管理 + Agent 委托问答。"""
import json
import os
from openai import OpenAI

from src.common.pdf_parser import extract_text
from src.common.embedding import EmbeddingModel
from src.common.dual_index import DualIndex
from src.rag.chunker import split_text, split_text_with_sections
from src.rag.abstract_extractor import extract_abstract
from src.rag.retriever import Retriever
from src.rag.agent import AgenticRAG


def _pdf_to_article_name(pdf_path: str) -> str:
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    return basename.replace(" ", "_")


class RAGPipeline:
    """RAG 完整流程：摘要/正文双索引 + Agent 路由问答。"""

    def __init__(self, config: dict):
        self.config = config

        llm_cfg = config["llm"]
        client_kwargs = {"api_key": llm_cfg["api_key"]}
        if llm_cfg.get("base_url"):
            client_kwargs["base_url"] = llm_cfg["base_url"]
        self.client = OpenAI(**client_kwargs)

        emb_cfg = config["embedding"]
        self.embedding_model = EmbeddingModel(
            api_key=emb_cfg["api_key"],
            model=emb_cfg["model"],
            base_url=emb_cfg.get("base_url"),
        )
        self.dim = emb_cfg.get("dimension", 2048)

        self.abstract_index = DualIndex(self.embedding_model, self.dim)
        self.chunk_index = DualIndex(self.embedding_model, self.dim)

        self._meta: dict = {"total_chunks": 0, "articles": []}
        self.retriever: Retriever | None = None
        self.agent: AgenticRAG | None = None

    # ═══════════════════════════════════════════════════════
    @property
    def index_dir(self) -> str:
        return self.config["index"]["index_dir"]

    @property
    def chunks_dir(self) -> str:
        return self.config["index"]["chunks_dir"]

    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.json")

    def _chunks_file(self, article_name: str) -> str:
        return os.path.join(self.chunks_dir, f"{article_name}_chunks.json")

    def _abstract_file(self, article_name: str) -> str:
        return os.path.join(self.chunks_dir, f"{article_name}_abstract.json")

    def _faiss_dir(self, kind: str) -> str:
        return os.path.join(self.index_dir, f"{kind}_faiss")

    def _bm25_path(self, kind: str) -> str:
        return os.path.join(self.index_dir, f"{kind}_bm25.pkl")

    # ─── meta ──────────────────────────────────────────────
    def _read_meta(self) -> dict:
        path = self._meta_path()
        if not os.path.exists(path):
            return {"total_chunks": 0, "articles": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_meta(self, meta: dict):
        os.makedirs(self.index_dir, exist_ok=True)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self._meta = meta

    # ═══════════════════════════════════════════════════════
    #  构建索引
    # ═══════════════════════════════════════════════════════
    def build_index(self, pdf_path: str):
        print("=" * 60)
        print(f"构建索引: {pdf_path}")
        print("=" * 60)
        article_name = _pdf_to_article_name(pdf_path)

        text = extract_text(pdf_path)
        if not text:
            raise ValueError("PDF 未提取到有效文本")

        preamble, sections = split_text_with_sections(text)
        abstract = extract_abstract(text, preamble, sections)
        abstract["article"] = article_name
        new_chunks = split_text(text)
        print(
            f"文章: {article_name} | 标题: {abstract['title'][:60]} | "
            f"摘要 {len(abstract['content'])} 字 | {len(new_chunks)} 个块"
        )

        meta = self._read_meta()
        # 移除旧记录
        meta["articles"] = [a for a in meta["articles"] if a["name"] != article_name]
        for old in (self._chunks_file(article_name), self._abstract_file(article_name)):
            if os.path.exists(old):
                os.remove(old)

        # 写文件
        os.makedirs(self.chunks_dir, exist_ok=True)
        with open(self._chunks_file(article_name), "w", encoding="utf-8") as f:
            json.dump(new_chunks, f, ensure_ascii=False, indent=2)
        with open(self._abstract_file(article_name), "w", encoding="utf-8") as f:
            json.dump(abstract, f, ensure_ascii=False, indent=2)

        meta["articles"].append({
            "name": article_name,
            "chunks_file": f"{article_name}_chunks.json",
            "abstract_file": f"{article_name}_abstract.json",
            "chunk_count": len(new_chunks),
            "chunk_range": None,  # 重排时填
        })

        self._rebuild_all(meta)

    def _rebuild_all(self, meta: dict):
        """读所有文章的 abstract/chunks，重建双索引并写 meta。"""
        articles = meta.get("articles", [])

        all_abstract_texts: list[str] = []
        all_chunk_texts: list[str] = []
        cursor = 0
        for art in articles:
            with open(
                os.path.join(self.chunks_dir, art["abstract_file"]),
                "r", encoding="utf-8",
            ) as f:
                ab = json.load(f)
            ab_text = f"{ab.get('title', '')}\n{ab.get('content', '')}".strip()
            all_abstract_texts.append(ab_text)

            with open(
                os.path.join(self.chunks_dir, art["chunks_file"]),
                "r", encoding="utf-8",
            ) as f:
                chunks = json.load(f)
            for c in chunks:
                all_chunk_texts.append(c["content"])
            art["chunk_count"] = len(chunks)
            art["chunk_range"] = [cursor, cursor + len(chunks) - 1] if chunks else [cursor, cursor - 1]
            cursor += len(chunks)

        meta["total_chunks"] = cursor
        meta["total_articles"] = len(articles)
        self._write_meta(meta)

        print(
            f"  重建索引: {len(articles)} 篇摘要, {cursor} 个 chunk..."
        )
        self.abstract_index.build(all_abstract_texts)
        self.chunk_index.build(all_chunk_texts)
        print("索引构建完成！")

    def save_index(self):
        os.makedirs(self.index_dir, exist_ok=True)
        print("保存索引...")
        self.abstract_index.save(self._faiss_dir("abstract"), self._bm25_path("abstract"))
        self.chunk_index.save(self._faiss_dir("chunk"), self._bm25_path("chunk"))
        print(f"索引已保存到 {self.index_dir}/")

    # ═══════════════════════════════════════════════════════
    #  加载
    # ═══════════════════════════════════════════════════════
    def load_index(self):
        meta = self._read_meta()
        if not meta or not meta.get("articles"):
            raise FileNotFoundError(
                "索引不存在，请先执行 python main.py index <pdf_path>"
            )

        self.abstract_index.load(
            self._faiss_dir("abstract"),
            self._bm25_path("abstract"),
            size=len(meta["articles"]),
        )
        self.chunk_index.load(
            self._faiss_dir("chunk"),
            self._bm25_path("chunk"),
            size=meta["total_chunks"],
        )
        self._meta = meta
        self._build_runtime()

        print("=" * 50)
        print(
            f"已加载索引：{meta['total_chunks']} 个 chunk，"
            f"{len(meta['articles'])} 篇文章"
        )

    def _build_runtime(self):
        self.retriever = Retriever(
            abstract_index=self.abstract_index,
            chunk_index=self.chunk_index,
            meta=self._meta,
            chunks_dir=self.chunks_dir,
        )
        self.agent = AgenticRAG(
            retriever=self.retriever,
            llm_client=self.client,
            llm_model=self.config["llm"]["model"],
            retrieval_cfg=self.config["retrieval"],
        )

    # ═══════════════════════════════════════════════════════
    #  文章管理
    # ═══════════════════════════════════════════════════════
    def list_articles(self) -> list[str]:
        meta = self._read_meta()
        return [a["name"] for a in meta.get("articles", [])]

    def delete_article(self, article_name: str):
        meta = self._read_meta()
        if not any(a["name"] == article_name for a in meta["articles"]):
            print(f"未找到文章 '{article_name}'")
            return

        meta["articles"] = [a for a in meta["articles"] if a["name"] != article_name]
        for old in (self._chunks_file(article_name), self._abstract_file(article_name)):
            if os.path.exists(old):
                os.remove(old)

        print(f"已移除文章: {article_name}，重建双索引...")
        self._rebuild_all(meta)
        self.save_index()
        print("删除完成")

    # ═══════════════════════════════════════════════════════
    #  问答（委托 Agent）
    # ═══════════════════════════════════════════════════════
    def query(self, question: str) -> tuple[str, list[dict]]:
        if self.agent is None:
            self._build_runtime()
        return self.agent.answer(question)
