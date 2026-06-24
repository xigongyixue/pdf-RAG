#!/bin/bash
# PDF-RAG 使用说明 & 运行实例

# ─── 安装依赖 ─────────────────────────────────────────────
pip install -r requirements.txt


# ─── 1. 索引管理 ─────────────────────────────────────────

# 添加文章到索引（每篇都会生成 _abstract.json + _chunks.json，重建双索引）
python main.py index "pdf/AI meets database AI4DB and DB4AI.pdf"
python main.py index "pdf/AdaSlice.pdf"
python main.py index "pdf/Bf-Tree A Modern Read-Write-Optimized Concurrent Larger-Than-Memory Range Index.pdf"
python main.py index "pdf/Write-Aware Timestamp Tracking Effective and Efficient Page Replacement for Modern Hardware.pdf"

# 列出已索引文章
python main.py list

# 删除某文章（双索引同步重建）
python main.py delete AdaSlice


# ─── 2. Agentic RAG 三类查询示例 ─────────────────────────

# 策略 A: ABSTRACT_FIRST —— 主题/方法定位类
# 先用摘要锁定相关文章，再到这些文章的正文中提取答案
python main.py query "AI4DB 在查询优化中有哪些应用？"

# 策略 B: DIRECT_CHUNK —— 跨文章特征枚举类
# 答案分散在多篇文章里，直接全库检索 chunk
python main.py query "哪些方法使用了 LRU 缓存？"

# 策略 C: MULTI_HOP —— 对比/多跳类
# 拆成子问题并行检索，每个子问题走 ABSTRACT_FIRST 后聚合
python main.py query "Bf-Tree 和传统 B+Tree 的区别是什么？"


# ─── 3. 交互式模式 ───────────────────────────────────────
python main.py interactive


# ─── 4. 测试分块效果 ─────────────────────────────────────
python test.py "pdf/AdaSlice.pdf"


# ─── 5. MCP 服务（被 Claude Code 等 MCP 客户端调用） ─────
# 通过 mcp_config.json 注册到客户端后，会暴露 4 个工具：
#   pdf_rag_list / pdf_rag_index / pdf_rag_delete / pdf_rag_query
# pdf_rag_query 返回中包含完整 chunk 内容
python mcp_server.py
