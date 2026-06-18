"""查询翻译模块：将中文查询翻译为英文以提升检索效果。"""
from openai import OpenAI


def translate_query(client: OpenAI, query: str, model: str = "gpt-4o-mini") -> str:
    """将中文查询翻译为英文。

    Args:
        client: OpenAI 客户端
        query: 中文查询文本
        model: 使用的模型名称

    Returns:
        翻译后的英文查询
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a translator. Translate the following Chinese query into English for use in a document retrieval system. "
                           "Output ONLY the English translation, nothing else. Do not add any explanations or notes.",
            },
            {"role": "user", "content": query},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()
