"""摘要提取模块：基于规则从论文 PDF 文本中抽取标题与摘要。"""
import re


_MAX_ABSTRACT_CHARS = 1500
_FALLBACK_CHARS = 800


def extract_abstract(text: str, preamble: str, sections: list[tuple[str, str]]) -> dict:
    """从论文文本中规则化提取 {title, content}。

    Args:
        text: PDF 全文（包含 preamble）
        preamble: chunker 切出的 preamble 段（首章节之前）
        sections: chunker 切出的章节列表

    Returns:
        {"title": str, "content": str}
    """
    return {
        "title": _extract_title(text),
        "content": _extract_abstract_body(preamble, sections),
    }


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if 3 <= len(line) <= 200:
            return line
        return line[:200]
    return "UNTITLED"


def _extract_abstract_body(preamble: str, sections: list[tuple[str, str]]) -> str:
    body = _find_abstract_in_preamble(preamble)
    if body:
        return body[:_MAX_ABSTRACT_CHARS].strip()

    if preamble.strip():
        return preamble.strip()[-_FALLBACK_CHARS:]

    if sections:
        _, first_body = sections[0]
        return first_body.strip()[:_FALLBACK_CHARS]

    return ""


_ABSTRACT_HEADER = re.compile(r'(?im)^\s*abstract\b[:\.\s]*')
_ABSTRACT_END = re.compile(
    r'(?im)^\s*(?:keywords?|index\s+terms|1\s*\.?\s*introduction|introduction)\b'
)


def _find_abstract_in_preamble(preamble: str) -> str:
    if not preamble:
        return ""

    header_match = _ABSTRACT_HEADER.search(preamble)
    if not header_match:
        return ""

    tail = preamble[header_match.end():]
    end_match = _ABSTRACT_END.search(tail)
    body = tail[:end_match.start()] if end_match else tail
    return body.strip()
