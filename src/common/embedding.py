"""向量嵌入模块：使用火山引擎 Ark (Doubao) 将文本转换为向量。"""
from volcenginesdkarkruntime import Ark


class EmbeddingModel:
    """火山引擎 Ark 向量嵌入模型封装。使用 multimodal_embeddings API。"""

    def __init__(
        self,
        api_key: str,
        model: str = "doubao-embedding-vision-251215",
        base_url: str | None = None,
    ):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = Ark(**client_kwargs)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """对文本列表生成向量嵌入。批量处理多个文本以提高效率。"""
        # 构建批量输入
        inputs = [{"text": text, "type": "text"} for text in texts]
        
        # 批量调用 API
        resp = self.client.multimodal_embeddings.create(
            model=self.model,
            input=inputs,
        )
        
        # 处理不同的响应格式
        embeddings = []
        
        # 检查 resp.data 的类型
        if hasattr(resp.data, 'embedding'):
            # 单个结果的情况
            embeddings.append(resp.data.embedding)
        elif isinstance(resp.data, list):
            # 多个结果的情况
            for item in resp.data:
                if hasattr(item, 'embedding'):
                    embeddings.append(item.embedding)
                elif isinstance(item, list) and all(isinstance(x, (int, float)) for x in item):
                    embeddings.append(item)
                else:
                    # 其他情况，尝试直接使用
                    embeddings.append(item)
        else:
            # 其他情况，尝试直接使用 resp.data
            embeddings.append(resp.data)
        
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """对单条查询文本生成向量。"""
        return self.embed([text])[0]
