"""文本分块模块：按论文章节切分，每个章节作为一个整块。"""
import re


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 0) -> list[dict]:
    """按论文 section 切分，同一章节的内容保持为一个整体块。

    Args:
        text: 输入文本
        chunk_size: (保留参数，当前版本不使用)
        chunk_overlap: (保留参数，当前版本不使用)

    Returns:
        分块列表，每项为 {"index": int, "section": str, "content": str}
    """
    sections = _split_by_sections(text)

    all_chunks = []
    for section_title, section_body in sections:
        body = section_body.strip()
        if not body:
            continue
        all_chunks.append({
            "section": section_title,
            "content": body,
        })

    # 全局统一编号
    for i, chunk in enumerate(all_chunks):
        chunk["index"] = i

    return all_chunks


def _split_by_sections(text: str) -> list[tuple[str, str]]:
    """按学术论文章节编号将文本切分为 (section标题, section正文) 列表。

    识别模式：
      - "\n1\nINTRODUCTION\n"       (一级章节)
      - "\n2.1\nWhat Database ...\n" (二级章节)
      - "\n3.2.1\nSubsection\n"      (三级章节)

    通过后置过滤排除页码、公式编号、图表标号等误匹配。
    """
    # 宽松匹配：换行 + 数字编号 + 换行 + 大写字母开头
    pattern = re.compile(r'\n(\d+(?:\.\d+)*)\n(?=[A-Z])')

    parts = pattern.split(text)

    sections = []
    if parts[0].strip():
        sections.append(("PREAMBLE", parts[0]))

    for i in range(1, len(parts), 2):
        sep_num = parts[i].strip()
        if i + 1 < len(parts):
            body = parts[i + 1]
            title_text = body.split('\n')[0].strip()

            if not _is_valid_section(sep_num, title_text):
                # 非真实章节 → 合并到前一个 section
                if sections:
                    prev_title, prev_body = sections[-1]
                    merged_body = prev_body + "\n" + sep_num + "\n" + body
                    sections[-1] = (prev_title, merged_body)
                continue

            full_title = f"{sep_num} {title_text}" if title_text else sep_num
            sections.append((full_title, body))

    return sections


def _is_valid_section(num: str, title: str) -> bool:
    """判断是否是真实的章节标题（排除页码、公式编号、图表标号等）。"""
    if not title or len(title) < 3:
        return False

    # 章节编号不能是 0
    if num == '0':
        return False

    # 带点的子章节（如 2.1）几乎总是有效
    if '.' in num:
        return title[0].isupper() and len(title) >= 3

    # 单独数字：可能是章节标题，也可能是页码/图表编号
    try:
        n = int(num)
    except ValueError:
        return False

    # 页码通常很大（>20），论文正文不会超过20个一级章节
    if n > 20:
        return False

    # 章节标题通常全大写或至少60%大写字母
    alpha_chars = [c for c in title if c.isalpha()]
    if not alpha_chars:
        return False
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    return upper_ratio >= 0.6
