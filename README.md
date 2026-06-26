# PDF-RAG

基于 RAG（检索增强生成）的知识库系统，用于存储和检索 PDF 文件中的英文信息，支持中文提问。

## 技术栈

| 组件 | 选型 |
|------|------|
| PDF 解析 | PyMuPDF |
| 文本分块 | 按章节语义分块 |
| 向量嵌入 | 火山引擎 Ark Doubao（2048维） |
| 向量检索 | FAISS |
| 标量检索 | BM25 (rank-bm25) |
| 混合融合 | RRF (Reciprocal Rank Fusion) |
| LLM | DeepSeek |
| 查询翻译 | 中文 → 英文（LLM翻译后检索） |

## 架构流程

```
PDF → 文本提取 → 按章节分块 → Embedding → FAISS
                          → 分词    → BM25

中文查询 → 翻译为英文 → 向量+BM25混合检索 → RRF融合 → LLM生成中文回答
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml`，填写 API Key：

```yaml
llm:
  provider: "deepseek"
  api_key: "sk-your-key"
  base_url: "https://api.deepseek.com"
  model: "deepseek-v4-flash"

embedding:
  provider: "volcengine"
  api_key: "ark-your-key"
  model: "doubao-embedding-vision-251215"
  dimension: 2048

paths:
  index_dir: "./index"
  chunks_dir: "./chunks"
  pdf_dir: "./pdfs"
```

### 3. 构建索引

```bash
# 单篇 PDF 构建索引
python main.py index /path/to/document.pdf

# 多篇 PDF 构建索引
python main.py index /path/to/pdf1.pdf /path/to/pdf2.pdf
```

### 4. 问答

```bash
# 单次查询（中文提问）
python main.py query "这篇文档的主要内容是什么？"

# 交互式问答
python main.py interactive
```

**查询结果示例：**
```
问题：文档中提到的主要算法有哪些？

回答：文档中主要提到了以下算法：
1. 支持向量机（SVM）- 用于分类任务
2. 随机森林 - 用于回归和分类
3. 深度学习模型 - 用于复杂模式识别

引用来源：
- 文章 "machine_learning.pdf"，块索引：[3, 7, 12]
```

### 5. 管理知识库

```bash
# 列出已索引的文档
python main.py list

# 删除指定文档
python main.py delete "document_name.pdf"

# 清空所有索引
python main.py clear
```

## MCP 服务模式

### 启动 MCP 服务

```bash
python mcp_server.py
```

### MCP 工具列表

| 工具名称 | 功能 | 参数 |
|---------|------|------|
| `pdf_rag_index` | 索引 PDF 文件 | `pdf_path`: PDF 文件路径 |
| `pdf_rag_list` | 列出已索引文档 | 无 |
| `pdf_rag_delete` | 删除指定文档 | `article_name`: 文章名称 |
| `pdf_rag_query` | 查询知识库 | `question`: 问题（中文） |

### MCP 使用示例

```python
from mcp.client import connect

client = connect("http://localhost:8000")

# 索引文档
result = client.call_tool("pdf_rag_index", {"pdf_path": "/path/to/document.pdf"})

# 查询
result = client.call_tool("pdf_rag_query", {"question": "文档的主要结论是什么？"})
```


## 核心特性

1. **按章节分块**：确保同一章节内容在一个块中，保持语义完整性
2. **混合检索**：向量检索 + BM25 检索，RRF 融合排序
3. **中文支持**：支持中文提问，自动翻译为英文检索
4. **来源追踪**：回答时显示引用的文章名称和块索引
5. **按需加载**：查询时才加载对应块，减少内存占用
6. **多文档管理**：支持多篇 PDF 的索引和管理
7. **MCP 服务**：支持通过 MCP 协议远程调用

## 使用注意事项

1. PDF 文件应为英文文本内容
2. 图片内容会被忽略（仅提取文本）
3. 查询语言支持中文和英文
4. 首次构建索引可能需要较长时间（取决于 PDF 数量和大小）
5. 建议定期备份 `index` 和 `chunks` 目录
