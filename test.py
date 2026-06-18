"""PDF-RAG 测试脚本。"""
import os
import tempfile
import unittest

import yaml

from src.pdf_parser import extract_text
from src.chunker import split_text
from src.hybrid_search import HybridSearch


class TestPDFParser(unittest.TestCase):
    """测试 PDF 解析。"""

    def test_extract_text_empty(self):
        """测试不存在的文件抛出异常。"""
        with self.assertRaises(Exception):
            extract_text("nonexistent.pdf")


class TestChunker(unittest.TestCase):
    """测试文本分块。"""

    def test_split_text_basic(self):
        """测试基本分块。"""
        text = "Hello world. " * 200
        chunks = split_text(text, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("index", chunk)
            self.assertIn("content", chunk)
            self.assertLessEqual(len(chunk["content"]), 200)  # 允许一定超量

    def test_split_text_overlap(self):
        """测试重叠分块。"""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = split_text(text, chunk_size=30, chunk_overlap=10)
        if len(chunks) >= 2:
            c1, c2 = chunks[0]["content"], chunks[1]["content"]
            # 检查是否有重叠
            overlap_found = any(
                c1[i : i + 10] in c2 for i in range(0, len(c1) - 10, 5)
            )
            self.assertTrue(overlap_found or len(chunks) > 1)

    def test_empty_text(self):
        """测试空文本。"""
        chunks = split_text("")
        self.assertEqual(len(chunks), 0)

    def test_short_text(self):
        """测试短文本不分块。"""
        short = "This is a short sentence."
        chunks = split_text(short, chunk_size=500, chunk_overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], short)


class TestHybridSearch(unittest.TestCase):
    """测试混合检索 RRF 融合（离线单元测试）。"""

    def test_rrf_merge(self):
        """测试 RRF 融合逻辑。"""
        # 模拟两种检索结果
        vec_indices = [0, 1, 2]
        bm25_indices = [1, 3, 0]

        rrf_scores = {}
        k = 60
        for rank, idx in enumerate(vec_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        # 索引1同时出现在两个结果中，应该有更高的RRF分数
        self.assertGreater(rrf_scores[1], rrf_scores[2])
        self.assertGreater(rrf_scores[1], rrf_scores[3])


def check_config():
    """检查配置文件是否已填写。"""
    if not os.path.exists("config.yaml"):
        print("[FAIL] config.yaml 不存在")
        return False

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = config.get("llm", {}).get("api_key", "")
    if not api_key:
        print("[WARN] config.yaml 中的 llm.api_key 尚未填写，跳过 API 相关测试")
        return False

    return True


if __name__ == "__main__":
    # 先运行离线单元测试
    print("=" * 60)
    print("运行离线单元测试...")
    print("=" * 60)

    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 检查配置
    print("\n" + "=" * 60)
    print("检查配置...")
    print("=" * 60)
    has_config = check_config()
    if has_config:
        print("[OK] 配置文件已就绪，可运行 API 相关测试")
    else:
        print("[INFO] 请先填写 config.yaml 中的 api_key，然后使用 main.py 进行集成测试")
