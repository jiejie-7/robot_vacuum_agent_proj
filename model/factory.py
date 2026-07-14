from abc import ABC
from abc import abstractmethod
from langchain_community.chat_models.tongyi import ChatTongyi, BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from typing import Optional
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def __init__(self, config_key: str = "chat_model_name"):
        self.config_key = config_key

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        model_name = str(rag_conf.get(self.config_key, "") or "").strip()
        if not model_name:
            return None
        return ChatTongyi(model=model_name)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory("chat_model_name").generator()
judge_model = ChatModelFactory("judge_model_name").generator()
embed_model = EmbeddingsFactory().generator()
