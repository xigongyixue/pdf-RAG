"""PDF-RAG 入口脚本。支持构建索引、列表、删除和问答。

用法:
  python main.py index <pdf_path>           # 添加文章到索引
  python main.py list                        # 列出已索引文章
  python main.py delete <article_name>       # 删除某文章
  python main.py query "<问题>"              # 跨文章问答
  python main.py interactive                 # 交互式跨文章问答
"""
import sys
import yaml

from src.rag.rag_pipeline import RAGPipeline


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_index(pipeline: RAGPipeline, pdf_path: str):
    """构建索引命令。"""
    pipeline.build_index(pdf_path)
    pipeline.save_index()


def cmd_list(pipeline: RAGPipeline):
    """列出已索引文章。"""
    articles = pipeline.list_articles()
    if not articles:
        print("暂无已索引的文章")
    else:
        print("已索引文章:")
        for name in articles:
            print(f"  - {name}")


def cmd_delete(pipeline: RAGPipeline, article_name: str):
    """删除某文章索引。"""
    articles = pipeline.list_articles()
    if not articles:
        print("暂无已索引的文章")
        return

    if article_name == "--select":
        print("选择要删除的文章:")
        for i, name in enumerate(articles):
            print(f"  [{i}] {name}")
        while True:
            try:
                choice = input("请选择序号: ").strip()
                article_name = articles[int(choice)]
                break
            except (ValueError, IndexError):
                print(f"无效序号，请输入 0-{len(articles)-1}")
    elif article_name not in articles:
        print(f"文章 '{article_name}' 不在索引列表中。可用文章: {articles}")
        return

    pipeline.delete_article(article_name)


def cmd_query(pipeline: RAGPipeline, question: str):
    """单次查询命令。"""
    pipeline.load_index()
    answer, sources = pipeline.query(question)

    print(f"\n{'='*60}")
    print(f"回答:\n{answer}")
    _print_sources(sources)
    print(f"{'='*60}")


def cmd_interactive(pipeline: RAGPipeline):
    """交互式问答命令。"""
    pipeline.load_index()
    print("\n交互式问答模式 | 输入 'quit' 退出\n")

    while True:
        try:
            question = input("请输入问题: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if not question:
                continue

            print()
            answer, sources = pipeline.query(question)
            print(f"\n回答:\n{answer}")
            _print_sources(sources)
            print()
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n再见！")
            break


def _print_sources(sources: list[dict], preview_chars: int = 200):
    """打印引用来源（含 chunk 内容预览）。"""
    if not sources:
        return
    by_article: dict[str, list[dict]] = {}
    for s in sources:
        by_article.setdefault(s["article"], []).append(s)

    print(f"\n引用来源 ({len(sources)} 个块):")
    for article, items in by_article.items():
        print(f"[{article}]")
        for it in items:
            content = (it.get("content") or "").strip().replace("\n", " ")
            if len(content) > preview_chars:
                content = content[:preview_chars] + "..."
            print(f"块[{it['index']:>3}] → {it['section']}")
            print(f"  {content}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config = load_config()

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("用法: python main.py index <pdf_path>")
            sys.exit(1)
        pipeline = RAGPipeline(config)
        cmd_index(pipeline, sys.argv[2])

    elif command == "list":
        pipeline = RAGPipeline(config)
        cmd_list(pipeline)

    elif command == "delete":
        article_name = sys.argv[2] if len(sys.argv) > 2 else "--select"
        pipeline = RAGPipeline(config)
        cmd_delete(pipeline, article_name)

    elif command == "query":
        if len(sys.argv) < 3:
            print('用法: python main.py query "<问题>"')
            sys.exit(1)
        pipeline = RAGPipeline(config)
        cmd_query(pipeline, sys.argv[2])

    elif command == "interactive":
        pipeline = RAGPipeline(config)
        cmd_interactive(pipeline)

    else:
        print(f"未知命令: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
