from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os

logger = logging.getLogger(__name__)

#获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
#建模型绝对路径
local_model_path = os.path.join(current_dir, "model", "Baai")

class LocalEmbeddingService:
    def __init__(self):
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model not found at {local_model_path}")

        logger.info(f"Loading local model from {local_model_path}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name = local_model_path,
            model_kwargs = {"device" : "cpu"},
            encode_kwargs = {"normalize_embeddings" : True}
        )
        logger.info("Embeddings model loaded successfully")

    def embed_text(self, text : str) -> list[float]:
        #对单段文本生成embedding向量
        vector = self.embeddings.embed_query(text)
        return vector

    def embed_documents(self, texts : list[str]) -> list[list[float]]:
        #对多段文本批量生成embeddings向量
        vectors = self.embeddings.embed_documents(texts)
        return vectors