#!/bin/bash
# PDF-RAG 使用说明

# 安装依赖
pip install -r requirements.txt

# 添加文章到索引
python main.py index /path/to/your.pdf

# 列出已索引文章
python main.py list

# 删除某文章
python main.py delete <article_name>

# 跨文章问答（无需指定文章名）
python main.py query "你的问题"

# 交互式跨文章问答
python main.py interactive

# 测试分块效果
python test.py /path/to/any.pdf
