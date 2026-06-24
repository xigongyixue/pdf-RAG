"""文本分块模块：按论文章节切分，每个章节作为一个整块。"""
import re


def split_text(text: str) -> list[dict]:
    """按论文 section 切分，同一章节的内容保持为一个整体块。

    Returns:
        分块列表，每项为 {"index": int, "section": str, "content": str}
    """
    preamble, sections = split_text_with_sections(text)

    all_chunks = []
    if preamble.strip():
        all_chunks.append({"section": "PREAMBLE", "content": preamble.strip()})

    for section_title, section_body in sections:
        body = section_body.strip()
        if not body:
            continue
        all_chunks.append({"section": section_title, "content": body})

    for i, chunk in enumerate(all_chunks):
        chunk["index"] = i

    return all_chunks


def split_text_with_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """切分文本为 (preamble, [(section_title, section_body), ...])。

    preamble 是第一个章节出现之前的所有内容（通常含标题、作者、摘要）。
    """
    pattern = re.compile(r'\n(\d+(?:\.\d+)*)\n(?=[A-Z])')
    parts = pattern.split(text)

    preamble = parts[0] if parts else ""
    sections: list[tuple[str, str]] = []

    for i in range(1, len(parts), 2):
        sep_num = parts[i].strip()
        if i + 1 >= len(parts):
            continue
        body = parts[i + 1]
        title_text = body.split('\n')[0].strip()

        if not _is_valid_section(sep_num, title_text):
            if sections:
                prev_title, prev_body = sections[-1]
                sections[-1] = (prev_title, prev_body + "\n" + sep_num + "\n" + body)
            else:
                preamble = preamble + "\n" + sep_num + "\n" + body
            continue

        full_title = f"{sep_num} {title_text}" if title_text else sep_num
        sections.append((full_title, body))

    return preamble, sections


def _is_valid_section(num: str, title: str) -> bool:
    if not title or len(title) < 3:
        return False
    if num == '0':
        return False
    if '.' in num:
        return title[0].isupper() and len(title) >= 3
    try:
        n = int(num)
    except ValueError:
        return False
    if n > 20:
        return False
    alpha_chars = [c for c in title if c.isalpha()]
    if not alpha_chars:
        return False
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    return upper_ratio >= 0.6
