"""PDF-RAG 测试脚本：分块效果验证（不涉及 embedding API）。"""
import json
import os
import yaml

from src.common.pdf_parser import extract_text
from src.rag.chunker import split_text


CHUNKS_OUTPUT_DIR = "chunks_output"


def test_chunking(pdf_path: str):
    """对 PDF 进行分块，存入 chunks_output/<文件名>_chunks.json，并展示切割效果。"""
    print("=" * 70)
    print(f"PDF: {pdf_path}")
    print("=" * 70)

    text = extract_text(pdf_path)
    chunks = split_text(text)

    print(f"\n总文本长度: {len(text)} 字符")
    print(f"分块数量:   {len(chunks)}\n")

    for c in chunks:
        content_len = len(c["content"])
        print(f"[{c['index']:>3}] {c['section']}  ({content_len} 字符)")

    # 输出目录 + 文件名：以 PDF 文件名为前缀
    os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_name = pdf_basename.replace(" ", "_")
    output_file = os.path.join(CHUNKS_OUTPUT_DIR, f"{safe_name}_chunks.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n所有分块已保存到 {output_file}")


def check_config():
    """检查配置文件是否存在。"""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"LLM:      {config['llm']['provider']} / {config['llm']['model']}")
        print(f"Embedding: {config['embedding']['provider']} / {config['embedding']['model']} ({config['embedding']['dimension']}维)")
        print("[OK] config.yaml 已就绪")
    except Exception as e:
        print(f"[WARN] config.yaml: {e}")


if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("PDF-RAG 分块测试")
    print("=" * 70)
    check_config()

    # 支持命令行指定文件: python test.py path/to/file.pdf
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.isfile(pdf_path):
            print(f"\n[ERROR] 文件不存在: {pdf_path}")
            sys.exit(1)
        if not pdf_path.endswith(".pdf"):
            print(f"\n[ERROR] 不是 PDF 文件: {pdf_path}")
            sys.exit(1)
        print()
        test_chunking(pdf_path)
        sys.exit(0)

    # 未指定文件时，自动从 pdf/ 目录查找
    pdf_dir = "pdf"
    if os.path.isdir(pdf_dir):
        pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
        if pdfs:
            pdf_path = os.path.join(pdf_dir, pdfs[0])
            print()
            test_chunking(pdf_path)
        else:
            print("\n[INFO] pdf/ 目录下无 PDF 文件")
    else:
        print("\n[INFO] pdf/ 目录不存在，请放入 PDF 文件后重试")
