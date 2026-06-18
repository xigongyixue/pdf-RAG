"""RAG 主流程编排模块：整合索引构建和问答流程。"""
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


class RAGPipeline:
    """RAG 完整流程：PDF 索引构建 + 检索问答。"""

    def __init__(self, config: dict):
        self.config = config

        # 初始化 LLM 客户端（DeepSeek，OpenAI 兼容接口）
        llm_cfg = config["llm"]
        client_kwargs = {"api_key": llm_cfg["api_key"]}
        if llm_cfg.get("base_url"):
            client_kwargs["base_url"] = llm_cfg["base_url"]
        self.client = OpenAI(**client_kwargs)

        # 初始化向量嵌入模型（火山引擎 Ark）
        emb_cfg = config["embedding"]
        self.embedding_model = EmbeddingModel(
            api_key=emb_cfg["api_key"],
            model=emb_cfg["model"],
        )

        # 初始化各组件
        self.vector_store = FAISSVectorStore(dimension=emb_cfg.get("dimension", 2048))
        self.bm25_index = BM25Index()
        self.hybrid_search = HybridSearch(
            self.embedding_model, self.vector_store, self.bm25_index
        )

        self.chunks: list[dict] = []  # {"index": int, "content": str}

    def build_index(self, pdf_path: str):
        """从 PDF 文件构建索引。

        Args:
            pdf_path: PDF 文件路径
        """
        print(f"[1/4] 解析PDF: {pdf_path}")
        text = extract_text(pdf_path)
        if not text:
            raise ValueError("PDF 未提取到有效文本")

        print(f"[2/4] 文本分块...")
        self.chunks = split_text(
            text,
            chunk_size=self.config["pdf"]["chunk_size"],
            chunk_overlap=self.config["pdf"]["chunk_overlap"],
        )
        print(f"  生成 {len(self.chunks)} 个文本块")

        print(f"[3/4] 生成向量并构建FAISS索引...")
        chunk_texts = [c["content"] for c in self.chunks]
        vectors = self.embedding_model.embed(chunk_texts)
        # L2 归一化
        vectors = np.array(vectors)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vector_store.add_vectors(vectors)

        print(f"[4/4] 构建BM25索引...")
        self.bm25_index.build(chunk_texts)

        print("索引构建完成！")

    def save_index(self):
        """保存索引到磁盘。"""
        index_cfg = self.config["index"]
        os.makedirs("index", exist_ok=True)

        print("保存索引...")
        self.vector_store.save(index_cfg["faiss_index_path"])
        self.bm25_index.save(index_cfg["bm25_index_path"])

        with open(index_cfg["chunk_store_path"], "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"索引已保存到 {index_cfg['faiss_index_path']}, {index_cfg['bm25_index_path']}")

    def load_index(self):
        """从磁盘加载索引。"""
        index_cfg = self.config["index"]

        print("加载索引...")
        self.vector_store.load(index_cfg["faiss_index_path"])
        self.bm25_index.load(index_cfg["bm25_index_path"])

        with open(index_cfg["chunk_store_path"], "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print(f"已加载 {len(self.chunks)} 个文本块")

    def query(self, question: str) -> str:
        """基于 RAG 流程回答问题。

        Args:
            question: 用户提问（中文）

        Returns:
            模型生成的回答
        """
        retrieval_cfg = self.config["retrieval"]

        # 1. 翻译查询为英文
        print("  [翻译] 中文 → 英文")
        english_query = translate_query(
            self.client, question, model=self.config["llm"]["model"]
        )
        print(f"  英文查询: {english_query}")

        # 2. 混合检索
        print(f"  [检索] 混合检索中...")
        indices = self.hybrid_search.search(
            english_query,
            embedding_top_k=retrieval_cfg["embedding_top_k"],
            bm25_top_k=retrieval_cfg["bm25_top_k"],
            final_top_k=retrieval_cfg["final_top_k"],
        )

        if not indices:
            return "未找到相关文档内容。"

        # 3. 获取上下文
        context_parts = []
        for idx in indices:
            context_parts.append(self.chunks[idx]["content"])

        context = "\n\n---\n\n".join(context_parts)

        # 4. 生成回答
        print(f"  [生成] 调用LLM生成回答...")
        response = self.client.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's question based on the provided context. "
                        "The context is extracted from PDF documents in English. "
                        "You MUST answer in Chinese (中文). "
                        "If the context does not contain enough information, say '根据提供的文档无法回答该问题'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content
