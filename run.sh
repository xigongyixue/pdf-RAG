#!/bin/bash
# PDF-RAG 使用说明

# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建索引（替换为你的PDF路径）
python main.py index /path/to/your.pdf

# 3. 问答模式
# 3.1 单次查询
python main.py query "这篇文档的主要内容是什么？"

# 3.2 交互式问答
python main.py interactive
