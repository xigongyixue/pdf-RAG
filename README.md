# PDF-RAG

基于 RAG（检索增强生成）的知识库系统，用于存储和检索 PDF 文件中的英文信息，支持中文提问。

## 技术栈

| 组件 | 选型 |
|------|------|
| PDF 解析 | PyMuPDF |
| 文本分块 | LangChain RecursiveCharacterTextSplitter |
| 向量嵌入 | 火山引擎 Ark Doubao（2048维） |
| 向量检索 | FAISS |
| 标量检索 | BM25 (rank-bm25) |
| 混合融合 | RRF (Reciprocal Rank Fusion) |
| LLM | DeepSeek |
| 查询翻译 | 中文 → 英文（LLM翻译后检索） |

## 架构流程

```
PDF → 文本提取 → 分块 → Embedding → FAISS
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
```

### 3. 构建索引

```bash
python main.py index /path/to/document.pdf
```

### 4. 问答

```bash
# 单次查询
python main.py query "这篇文档的主要内容是什么？"

# 交互式问答
python main.py interactive
```

### 5. 运行测试

```bash
python test.py
```

## 项目结构

```
pdf-RAG/
├── config.yaml          # 配置文件
├── main.py              # 入口脚本
├── test.py              # 测试脚本
├── requirements.txt     # Python 依赖
├── run.sh               # 使用说明脚本
└── src/
    ├── pdf_parser.py    # PDF 文本提取
    ├── chunker.py       # 文本分块
    ├── embedding.py     # 向量嵌入
    ├── vector_store.py  # FAISS 向量存储
    ├── bm25_index.py    # BM25 标量索引
    ├── translator.py    # 查询翻译
    ├── hybrid_search.py # 混合检索 + RRF 融合
    └── rag_pipeline.py  # RAG 主流程编排
```
