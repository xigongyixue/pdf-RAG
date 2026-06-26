"""文本分块模块：按论文章节切分，支持动态合并和切分。"""
import re


def split_text(
    text: str,
    min_chunk_size: int = 2000,
    max_chunk_size: int = 4000,
    overlap_ratio: float = 0.2,
) -> list[dict]:
    """按论文 section 切分，支持动态合并和切分。

    策略：
    1. 短章节合并：如果当前章节 < min_chunk_size，且与上一章节合并后 < max_chunk_size，则合并
    2. 长章节切分：如果章节 > max_chunk_size，切分为多个块，块间保留 overlap_ratio 重叠

    Args:
        text: 输入文本
        min_chunk_size: 最小块大小（字数）
        max_chunk_size: 最大块大小（字数）
        overlap_ratio: 长章节切分时的重叠比例

    Returns:
        分块列表，每项为 {"index": int, "section": str, "content": str}
    """
    preamble, sections = split_text_with_sections(text)

    all_chunks = []
    
    # 处理 preamble
    if preamble.strip():
        all_chunks.append({"section": "PREAMBLE", "content": preamble.strip()})

    # 处理各章节
    for section_title, section_body in sections:
        body = section_body.strip()
        if not body:
            continue
        all_chunks.append({"section": section_title, "content": body})

    # 动态合并和切分
    final_chunks = _dynamic_chunking(all_chunks, min_chunk_size, max_chunk_size, overlap_ratio)

    # 重新编号，并删除所有换行符
    for i, chunk in enumerate(final_chunks):
        chunk["section"] = _remove_newlines(chunk.get("section", ""))
        chunk["content"] = _remove_newlines(chunk.get("content", ""))
        chunk["index"] = i

    return final_chunks


def _remove_newlines(text: str) -> str:
    """删除文本中的所有换行符，并压缩多余空白。"""
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def _dynamic_chunking(
    chunks: list[dict],
    min_chunk_size: int,
    max_chunk_size: int,
    overlap_ratio: float,
) -> list[dict]:
    """对章节列表进行动态合并和切分。"""
    if not chunks:
        return []

    result: list[dict] = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]
        content = chunk["content"]
        section = chunk["section"]
        content_len = len(content)

        # 情况1: 长章节，需要切分
        if content_len > max_chunk_size:
            sub_chunks = _split_long_chunk(chunk, max_chunk_size, overlap_ratio)
            result.extend(sub_chunks)
            i += 1
            continue

        # 情况2: 短章节，尝试与前一个合并
        if content_len < min_chunk_size and result:
            prev_chunk = result[-1]
            prev_content = prev_chunk["content"]
            combined_content = prev_content + "\n\n" + content
            combined_len = len(combined_content)

            if combined_len <= max_chunk_size:
                # 合并成功
                prev_chunk["content"] = combined_content
                prev_chunk["section"] = prev_chunk["section"] + " / " + section
                i += 1
                continue
            else:
                # 合并后超出限制，保留当前块（不合并）
                result.append({"section": section, "content": content})
                i += 1
                continue

        # 情况3: 正常大小的章节，直接添加
        result.append({"section": section, "content": content})
        i += 1

    return result


def _split_long_chunk(
    chunk: dict,
    max_chunk_size: int,
    overlap_ratio: float,
) -> list[dict]:
    """将长章节切分为多个重叠的块。"""
    content = chunk["content"]
    section = chunk["section"]
    
    # 计算步长（每次移动的字符数）
    step = int(max_chunk_size * (1 - overlap_ratio))
    if step <= 0:
        step = max_chunk_size // 2  # 至少移动一半

    sub_chunks = []
    start = 0
    
    while start < len(content):
        end = start + max_chunk_size
        
        # 如果是最后一块，且剩余内容较少，直接包含剩余内容
        if end >= len(content) or (len(content) - end) < max_chunk_size // 4:
            end = len(content)
        else:
            # 在句子边界处切分
            end = _find_sentence_boundary(content, end)
        
        chunk_content = content[start:end].strip()
        if chunk_content:
            # 添加块编号到章节名
            chunk_num = len(sub_chunks) + 1
            sub_section = f"{section} (part {chunk_num})"
            sub_chunks.append({
                "section": sub_section,
                "content": chunk_content,
            })
        
        start += step

    return sub_chunks if sub_chunks else [{"section": section, "content": content}]


def _find_sentence_boundary(text: str, pos: int) -> int:
    """在指定位置附近查找句子边界。"""
    if pos >= len(text):
        return len(text)
    
    # 向前查找最近的句子结束符
    candidates = []
    for sep in ['.\n', '.\n', '!\n', '?\n', ';\n', ':\n', '\n\n']:
        idx = text.rfind(sep, max(0, pos - 100), pos + 10)
        if idx >= 0:
            candidates.append(idx + len(sep))
    
    if candidates:
        # 返回最接近 pos 的边界
        return min(candidates, key=lambda x: abs(x - pos))
    
    # 如果没找到句子边界，尝试找单词边界
    for sep in [' ', '\t', '\n']:
        idx = text.rfind(sep, max(0, pos - 50), pos + 10)
        if idx >= 0:
            return idx
    
    return pos


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
