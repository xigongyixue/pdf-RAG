"""RAG 主流程编排模块：meta.json + 独立 chunk json，跨文章检索。"""
import json
import os
import numpy as np
from openai import OpenAI

from src.pdf_parser import extract_text
from src.chunker import split_text
from src.embedding import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.bm25_index import BM25Index
from src.translator import translate_query
from src.hybrid_search import HybridSearch


def _pdf_to_article_name(pdf_path: str) -> str:
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    return basename.replace(" ", "_")


class RAGPipeline:
    """RAG 完整流程：meta.json 管理文章，独立 chunk json，跨文章检索。"""

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
        )

        dim = emb_cfg.get("dimension", 2048)
        self.vector_store = FAISSVectorStore(dimension=dim)
        self.bm25_index = BM25Index()
        self.hybrid_search = HybridSearch(
            self.embedding_model, self.vector_store, self.bm25_index
        )

        self.chunks: list[dict] = []         # 仅 build/delete 时用于重建
        self._meta: dict = {}                # meta.json 内容
        self._article_chunks_cache: dict[str, list[dict]] = {}  # 按需加载缓存

    @property
    def index_dir(self) -> str:
        return self.config["index"]["index_dir"]

    @property
    def chunks_dir(self) -> str:
        return self.config["index"]["chunks_dir"]

    # ═══════════════════════════════════════════════════════
    #  meta.json 读写
    # ═══════════════════════════════════════════════════════

    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "meta.json")

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
    #  chunks 读写
    # ═══════════════════════════════════════════════════════

    def _chunks_file(self, article_name: str) -> str:
        return os.path.join(self.chunks_dir, f"{article_name}_chunks.json")

    def _read_chunks(self, article_name: str) -> list[dict]:
        with open(self._chunks_file(article_name), "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_chunks(self, article_name: str, chunks: list[dict]):
        os.makedirs(self.chunks_dir, exist_ok=True)
        with open(self._chunks_file(article_name), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    # ═══════════════════════════════════════════════════════
    #  索引构建
    # ═══════════════════════════════════════════════════════

    def build_index(self, pdf_path: str):
        """从 PDF 构建索引，添加到统一索引中。"""
        print("=" * 60)
        print(f"构建索引: {pdf_path}")
        print("=" * 60)
        article_name = _pdf_to_article_name(pdf_path)

        # 1. 解析 + 分块
        text = extract_text(pdf_path)
        if not text:
            raise ValueError("PDF 未提取到有效文本")

        new_chunks = split_text(text)
        print(f"文章名: {article_name}，{len(new_chunks)} 个块")

        # 2. 读取已有 meta
        meta = self._read_meta()

        # 3. 如果文章已存在，先移除旧记录
        meta["articles"] = [a for a in meta["articles"] if a["name"] != article_name]
        # 删除旧 chunks 文件（如果存在）
        old_file = self._chunks_file(article_name)
        if os.path.exists(old_file):
            os.remove(old_file)

        # 4. 合并已有 chunks + 新 chunks，重新全局编号
        all_chunks = self._load_existing_chunks(meta)
        start_idx = len(all_chunks)
        for i, c in enumerate(new_chunks):
            c["article"] = article_name
            c["index"] = start_idx + i
        all_chunks.extend(new_chunks)

        # 5. 保存新文章的独立 json
        self._write_chunks(article_name, new_chunks)

        # 6. 更新 meta
        meta["total_chunks"] = len(all_chunks)
        meta["articles"].append({
            "name": article_name,
            "chunks_file": f"{article_name}_chunks.json",
            "chunk_range": [start_idx, start_idx + len(new_chunks) - 1],
            "chunk_count": len(new_chunks),
        })
        # 修复：meta 文件过大时不写入 content，保持轻量
        for a in meta["articles"]:
            a.pop("content", None)
        self._write_meta(meta)

        # 7. 重建 FAISS/BM25
        self.chunks = all_chunks
        print(f"  总块数: {len(all_chunks)}，重建索引...")
        self._rebuild_index()
        print("索引构建完成！")

    def _load_existing_chunks(self, meta: dict) -> list[dict]:
        """从 meta 中列出的各文章 json 加载已有 chunks。"""
        all_chunks = []
        for art in meta.get("articles", []):
            fpath = self._chunks_file(art["name"])
            if os.path.exists(fpath):
                chunks = self._read_chunks(art["name"])
                for c in chunks:
                    # 确保有 article 字段
                    if "article" not in c:
                        c["article"] = art["name"]
                all_chunks.extend(chunks)
        # 按全局索引排序
        for i, c in enumerate(all_chunks):
            c["index"] = i
        return all_chunks

    def _rebuild_index(self):
        """重建 FAISS 和 BM25 索引。"""
        self.vector_store = FAISSVectorStore(
            dimension=self.config["embedding"].get("dimension", 2048)
        )
        self.bm25_index = BM25Index()

        if not self.chunks:
            return

        chunk_texts = [c["content"] for c in self.chunks]
        vectors = self.embedding_model.embed(chunk_texts)
        vectors = np.array(vectors)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vector_store.add_vectors(vectors)
        self.bm25_index.build(chunk_texts)
        self.hybrid_search = HybridSearch(
            self.embedding_model, self.vector_store, self.bm25_index
        )

    def save_index(self):
        """保存 FAISS 和 BM25 索引到磁盘。"""
        os.makedirs(self.index_dir, exist_ok=True)
        print("保存索引...")
        self.vector_store.save(os.path.join(self.index_dir, "faiss"))
        self.bm25_index.save(os.path.join(self.index_dir, "bm25.pkl"))
        print(f"索引已保存到 {self.index_dir}/")

    # ═══════════════════════════════════════════════════════
    #  加载
    # ═══════════════════════════════════════════════════════

    def load_index(self):
        """加载索引元信息 + FAISS + BM25（不加载 chunks，按需查询）。"""
        meta = self._read_meta()
        if not meta or not meta.get("articles"):
            raise FileNotFoundError(
                "索引不存在，请先执行 python main.py index <pdf_path>"
            )

        self.vector_store.load(os.path.join(self.index_dir, "faiss"))
        self.bm25_index.load(os.path.join(self.index_dir, "bm25.pkl"))
        self._meta = meta
        self.chunks = []  # 不预加载
        self._article_chunks_cache.clear()
        self.hybrid_search = HybridSearch(
            self.embedding_model, self.vector_store, self.bm25_index
        )
        print("="*50)
        print(
            f"已加载 {meta['total_chunks']} 块的索引元信息，"
            f"覆盖 {len(meta['articles'])} 篇文章"
        )

    def _get_chunk(self, global_idx: int) -> dict:
        """按全局索引按需加载单个 chunk（带文章级缓存）。

        Args:
            global_idx: 全局块索引

        Returns:
            chunk dict {index, article, section, content}
        """
        # 从 meta 找到所属文章
        for art in self._meta.get("articles", []):
            lo, hi = art["chunk_range"]
            if lo <= global_idx <= hi:
                local_idx = global_idx - lo
                # 缓存：避免重复加载同一文章
                if art["name"] not in self._article_chunks_cache:
                    self._article_chunks_cache[art["name"]] = self._read_chunks(art["name"])
                chunks = self._article_chunks_cache[art["name"]]
                return chunks[local_idx]
        raise IndexError(f"全局索引 {global_idx} 不在 meta 中")

    # ═══════════════════════════════════════════════════════
    #  文章管理
    # ═══════════════════════════════════════════════════════

    def list_articles(self) -> list[str]:
        """列出已索引文章（从 meta.json 读取，不扫描目录）。"""
        meta = self._read_meta()
        return [a["name"] for a in meta.get("articles", [])]

    def delete_article(self, article_name: str):
        """删除某篇文章：删 chunks 文件 + 更新 meta + 重建索引。"""
        meta = self._read_meta()
        before = [a for a in meta["articles"] if a["name"] == article_name]
        if not before:
            print(f"未找到文章 '{article_name}'")
            return

        meta["articles"] = [a for a in meta["articles"] if a["name"] != article_name]

        # 删除文章 chunks 文件
        chunks_path = self._chunks_file(article_name)
        if os.path.exists(chunks_path):
            os.remove(chunks_path)

        # 重建全局 chunks
        removed = before[0]["chunk_count"]
        all_chunks = self._load_existing_chunks(meta)
        meta["total_chunks"] = len(all_chunks)
        self._write_meta(meta)

        self.chunks = all_chunks
        print(f"已移除 {removed} 个块（文章: {article_name}），总块数: {len(all_chunks)}")
        print("重建索引...")
        self._rebuild_index()
        self.save_index()
        print("删除完成")

    # ═══════════════════════════════════════════════════════
    #  问答
    # ═══════════════════════════════════════════════════════

    def query(self, question: str) -> tuple[str, list[dict]]:
        """跨文章 RAG 问答。

        Returns:
            (回答, [{article, index, section}, ...])
        """
        retrieval_cfg = self.config["retrieval"]

        english_query = translate_query(
            self.client, question, model=self.config["llm"]["model"]
        )
        print(f"英文查询: {english_query}")

        indices = self.hybrid_search.search(
            english_query,
            embedding_top_k=retrieval_cfg["embedding_top_k"],
            bm25_top_k=retrieval_cfg["bm25_top_k"],
            final_top_k=retrieval_cfg["final_top_k"],
        )

        if not indices:
            return "未找到相关文档内容。", []

        context_parts = []
        sources = []
        for idx in indices:
            chunk = self._get_chunk(idx)
            article = chunk.get("article", "?")
            section = chunk.get("section", "?")
            context_parts.append(
                f"[{idx}] <{article}> {section}:\n{chunk['content']}"
            )
            sources.append({"article": article, "index": idx, "section": section})

        context = "\n\n---\n\n".join(context_parts)

        response = self.client.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question "
                        "based on the provided context. "
                        "The context is extracted from PDF documents in English. "
                        "Each fragment is labeled [N] <ARTICLE> SECTION. "
                        "You MUST answer in Chinese (中文). "
                        "If the context does not contain enough information, "
                        "say '根据提供的文档无法回答该问题'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content, sources
