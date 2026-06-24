"""PDF 解析模块：提取 PDF 中的文本内容。"""
import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """从 PDF 文件中提取所有文本。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        提取到的全部文本，页之间用换行符分隔
    """
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        text = page.get_text()
        if text.strip():
            all_text.append(text.strip())

    doc.close()
    return "\n\n".join(all_text)
