"""向量嵌入模块：使用火山引擎 Ark (Doubao) 将文本转换为向量。"""
import numpy as np
from volcenginesdkarkruntime import Ark


class EmbeddingModel:
    """火山引擎 Ark 向量嵌入模型封装。使用 multimodal_embeddings API。"""

    def __init__(self, api_key: str, model: str = "doubao-embedding-vision-251215"):
        self.client = Ark(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """对文本列表生成向量嵌入。每次请求只处理一个文本。

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        embeddings = []
        for text in texts:
            resp = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[{"text": text, "type": "text"}],
            )
            embeddings.append(resp.data.embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """对单条查询文本生成向量。

        Args:
            text: 查询文本

        Returns:
            向量
        """
        return self.embed([text])[0]
