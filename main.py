"""PDF-RAG 入口脚本。支持构建索引和问答两种模式。

用法:
  python main.py index <pdf_path>           # 构建索引
  python main.py query "<问题>"             # 问答
  python main.py interactive                # 交互式问答
"""
import sys
import yaml

from src.rag_pipeline import RAGPipeline


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_index(pipeline: RAGPipeline, pdf_path: str):
    """构建索引命令。"""
    pipeline.build_index(pdf_path)
    pipeline.save_index()


def cmd_query(pipeline: RAGPipeline, question: str):
    """单次查询命令。"""
    pipeline.load_index()
    answer = pipeline.query(question)
    print(f"\n{'='*60}")
    print(f"回答:\n{answer}")
    print(f"{'='*60}")


def cmd_interactive(pipeline: RAGPipeline):
    """交互式问答命令。"""
    pipeline.load_index()
    print("\n交互式问答模式（输入 'quit' 退出）\n")

    while True:
        try:
            question = input("请输入问题: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("再见！")
                break
            if not question:
                continue

            print()
            answer = pipeline.query(question)
            print(f"\n回答:\n{answer}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n再见！")
            break


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config = load_config()
    pipeline = RAGPipeline(config)

    command = sys.argv[1]

    if command == "index":
        if len(sys.argv) < 3:
            print("用法: python main.py index <pdf_path>")
            sys.exit(1)
        pdf_path = sys.argv[2]
        cmd_index(pipeline, pdf_path)

    elif command == "query":
        if len(sys.argv) < 3:
            print('用法: python main.py query "<问题>"')
            sys.exit(1)
        question = sys.argv[2]
        cmd_query(pipeline, question)

    elif command == "interactive":
        cmd_interactive(pipeline)

    else:
        print(f"未知命令: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
