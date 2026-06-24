"""RRF (Reciprocal Rank Fusion) 融合函数：合并多路检索结果。"""


def rrf_fuse(
    rankings: list[list[int]],
    final_top_k: int,
    rrf_k: int = 60,
) -> list[int]:
    """对多路检索的索引排名做 RRF 融合，返回前 final_top_k 个全局索引。

    Args:
        rankings: 每路检索的索引列表（已按相关性降序）
        final_top_k: 融合后返回数量
        rrf_k: RRF 平滑参数

    Returns:
        按 RRF 分数降序的索引列表
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            if idx is None or idx < 0:
                continue
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_items[:final_top_k]]
