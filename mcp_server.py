"""PDF-RAG MCP 服务：暴露 index/list/delete/query 为 MCP tools。"""
import json
import asyncio
import yaml

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.rag_pipeline import RAGPipeline

# 全局 pipeline 实例（懒加载）
_pipeline: RAGPipeline | None = None
_config: dict | None = None


def _get_pipeline(load: bool = True) -> RAGPipeline:
    global _pipeline, _config
    if _pipeline is None:
        with open("config.yaml", "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        _pipeline = RAGPipeline(_config)
        if load:
            try:
                _pipeline.load_index()
            except FileNotFoundError:
                pass  # 尚未索引任何文章，后续 index 操作会创建
    return _pipeline


# ── MCP Server ────────────────────────────────────────────

server = Server("pdf-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="pdf_rag_list",
            description="列出所有已索引的 PDF 文章名称",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="pdf_rag_index",
            description="将 PDF 文件添加到知识库索引中",
            inputSchema={
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "PDF 文件的绝对路径",
                    },
                },
                "required": ["pdf_path"],
            },
        ),
        Tool(
            name="pdf_rag_delete",
            description="从知识库中删除某篇文章的索引",
            inputSchema={
                "type": "object",
                "properties": {
                    "article_name": {
                        "type": "string",
                        "description": "文章名称（可用 pdf_rag_list 查看）",
                    },
                },
                "required": ["article_name"],
            },
        ),
        Tool(
            name="pdf_rag_query",
            description="基于 RAG 知识库回答问题，返回回答内容和引用来源（文章名、块索引、章节名）",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "要查询的问题（中文或英文）",
                    },
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    pipeline = _get_pipeline()

    if name == "pdf_rag_list":
        articles = pipeline.list_articles()
        if not articles:
            return [TextContent(type="text", text="暂无已索引的文章")]
        result = "已索引文章:\n" + "\n".join(f"  - {a}" for a in articles)
        return [TextContent(type="text", text=result)]

    elif name == "pdf_rag_index":
        pdf_path = arguments["pdf_path"]
        pipeline = _get_pipeline(load=False)  # 先不加载，避免无索引时报错
        try:
            pipeline.build_index(pdf_path)
            pipeline.save_index()
            # 索引建立后，重置 pipeline 让下一次查询能加载
            global _pipeline
            _pipeline = None
            return [TextContent(type="text", text=f"索引构建成功: {pdf_path}")]
        except Exception as e:
            return [TextContent(type="text", text=f"索引构建失败: {e}")]

    elif name == "pdf_rag_delete":
        article_name = arguments["article_name"]
        try:
            pipeline.delete_article(article_name)
            return [TextContent(type="text", text=f"已删除文章: {article_name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"删除失败: {e}")]

    elif name == "pdf_rag_query":
        question = arguments["question"]
        try:
            answer, sources = pipeline.query(question)

            # 构建结构化来源
            source_lines = [f"# 回答\n\n{answer}\n\n# 引用来源"]
            by_article: dict[str, list[dict]] = {}
            for s in sources:
                by_article.setdefault(s["article"], []).append(s)

            for article, items in by_article.items():
                source_lines.append(f"\n## {article}")
                for it in items:
                    source_lines.append(
                        f"- 块[{it['index']}] → {it['section']}"
                    )

            # 附加 chunk 内容的简短摘要
            source_lines.append("\n# 引用的相关块内容片段")
            for s in sources:
                source_lines.append(
                    f"\n--- 块[{s['index']}] {s['section']} ---"
                )

            return [TextContent(type="text", text="\n".join(source_lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"查询失败: {e}")]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
