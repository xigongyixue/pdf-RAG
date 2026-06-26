# AGENTS.md

本文件为 AI 编码代理提供 PDF-RAG 项目的工作指南。

## 项目简介

基于 RAG 的论文知识库系统：解析 PDF（英文）→ 按章节动态分块 → 摘要/正文双索引（FAISS + BM25 + RRF）→ Agent 路由问答（中文）。数据持久化在 MySQL，向量/词法索引存于本地文件。

## 技术栈

| 组件 | 选型 |
|------|------|
| PDF 解析 | PyMuPDF |
| 文本分块 | 按章节语义分块 + 动态合并/切分 |
| 向量嵌入 | 火山引擎 Ark Doubao（2048 维，多向量批量） |
| 向量检索 | FAISS |
| 标量检索 | BM25 (rank-bm25) |
| 混合融合 | RRF (Reciprocal Rank Fusion) |
| LLM | DeepSeek（OpenAI 兼容接口） |
| 元数据存储 | MySQL（articles / abstracts / chunks 三张表） |

## 目录结构

```
main.py                      # CLI 入口：index / list / delete / delete-all / query / interactive
mcp_server.py                # MCP 服务模式
config.yaml                  # LLM / embedding / retrieval / index / database 配置
src/
  common/
    pdf_parser.py            # PDF 文本提取
    embedding.py             # Ark 向量嵌入（批量多向量）
    dual_index.py            # FAISS + BM25 + RRF 复合索引
    vector_store.py          # FAISS 封装
    bm25_index.py            # BM25 封装
    db_manager.py            # MySQL 管理（建库建表 + 增删改查 + 迁移）
  rag/
    chunker.py               # 章节切分 + 动态分块（合并短章节/切分长章节/去换行）
    abstract_extractor.py    # 摘要提取
    hybrid_search.py         # rrf_fuse 融合函数
    retriever.py             # 摘要检索 + chunk 检索（按文章聚合、章节排序）
    agent.py                 # 查询策略：单问题/多问题 + 追问上下文
    rag_pipeline.py          # 全流程编排：建索引/加载/删除/问答
index/                       # FAISS 索引 + BM25 pkl（meta 已迁移至 MySQL）
```

## 常用命令

```bash
python main.py index <pdf_path>        # 构建/更新某 PDF 的索引
python main.py list                    # 列出已索引文章（带数字索引）
python main.py delete <name|index>     # 删除文章（支持文章名或 list 数字索引）
python main.py delete-all              # 删除全部文章（需输入 DELETE ALL 二次确认）
python main.py query "<中文问题>"       # 单次问答
python main.py interactive             # 交互式问答（支持 clear 清空追问上下文）

# 语法检查（无测试框架时用于快速校验）
python -m py_compile <file.py>
```

## 查询策略（agent.py）

整体仅两类策略：

- `SINGLE_QUERY`（单问题）
  1. 用问题检索摘要，锁定相关论文
  2. 基于摘要重写问题，在锁定论文范围内检索 chunk
  3. 用原问题直接检索 chunk
  4. 合并 2、3 的结果（重复块保留高分，按文章聚合 + 章节顺序输出）
- `MULTI_QUERY`（多问题）
  - 拆分为子问题，并行执行单问题流程，再合并

追问：交互模式下 Agent 记住上一轮问答，自动将追问改写为独立问题；`clear`/`reset` 可清空上下文。

## 数据存储约定

- MySQL 三张表：`articles`、`abstracts`、`chunks`
- `chunks.global_index` 为全局唯一索引（UNIQUE 约束），跨文章连续分配
- 外键级联删除：删除文章会自动删除其摘要与 chunk
- 读取优先 MySQL，缺失时回退本地 JSON 文件
- `config.yaml` 的 `database` 段配置连接信息

## 代码规范

- 注释与文档使用中文，与现有风格保持一致
- 优先编辑现有文件，避免新建多余文件；不主动创建文档类文件（除非用户要求）
- 仅做被要求或明确必要的改动，避免过度设计
- Windows + PowerShell 环境：命令分隔用 `;`，不支持 `&&`
- 不要提交包含密钥的文件；`config.yaml` 当前含明文 API Key，改动时勿外泄

## 注意事项

- 修改 `chunker.py` 的分块策略或 `embedding.py` 的维度后，需重新 `index` 重建索引
- `dual_index.search` 与 `rrf_fuse` 返回 `(index, score)`，改动需同步上下游
- 删除/重建索引会调用 `_rebuild_all` 并重写 FAISS/BM25 文件
