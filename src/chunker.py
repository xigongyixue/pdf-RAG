"""文本分块模块：将文本按语义边界切分为固定大小的块。"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """将文本切分为多个块，每个块附带元数据。

    Args:
        text: 输入文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 相邻块之间的重叠字符数

    Returns:
        分块列表，每项为 {"index": int, "content": str}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)

    return [{"index": i, "content": chunk} for i, chunk in enumerate(chunks) if chunk.strip()]
