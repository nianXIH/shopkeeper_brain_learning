import threading
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from openai import OpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import FlagReranker
from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()

class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_client: Optional[ChatOpenAI]  = None
    _openai_llm_lock = threading.Lock()

    # json_object 模式的 LLM 客户端单独缓存：
    # 两种 response_format 若共用一个缓存槽，先创建的一方会覆盖另一方的配置，
    # 导致"是否输出 JSON"取决于节点执行顺序（非确定性行为）
    _openai_llm_json_client: Optional[ChatOpenAI] = None
    _openai_llm_json_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    # ── LLM / VLM ──

    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── LLM ──
    @classmethod
    def get_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        # 两种模式各自独立缓存，互不覆盖
        if response_format:
            return cls._get_or_create("_openai_llm_json_client", cls._openai_llm_json_lock,
                                      lambda: cls._create_llm_client(response_format))
        return cls._get_or_create("_openai_llm_client", cls._openai_llm_lock,
                                  lambda: cls._create_llm_client(response_format))

    @classmethod
    def _create_llm_client(cls, response_format) -> ChatOpenAI:
        try:

            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env('LLM_DEFAULT_MODEL')

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            llm_client = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                openai_api_key=api_key,
                openai_api_base=base_url,
                model_kwargs=model_kwargs
            )
            logger.info(f"OpenAI LLM 客户端初始化成功")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── BGE-M3嵌入模型客户端 ──
    @classmethod
    def get_bge_m3_client(cls):
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls):
        """
        创建bge_m3 客户端
        Returns:
        """

        try:
            # 1. 获取环境变量
            model_name = cls._require_env('BGE_M3_PATH')
            device = cls._require_env('BGE_DEVICE')
            fp16_str = cls._require_env('BGE_FP16')
            fp16 = fp16_str.lower() in ("true", "1")

            # 2. 创建
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败: {e}") from e


    # ── BGE-M3重排序模型客户端 ──
    @classmethod
    def get_bge_m3_rerank_client(cls):
        return cls._get_or_create("_bge_m3_rerank_client",
                                  cls._bge_m3_rerank_lock,
                                  cls._create_bge_m3_rerank_client)

    @classmethod
    def _create_bge_m3_rerank_client(cls):
        """
        创建bge_m3 重排序模型客户端
        Returns:
        """

        try:
            # 1. 获取环境变量
            model_name_or_path = cls._require_env('BGE_RERANKER_LARGE')
            device = cls._require_env('BGE_RERANKER_DEVICE')
            fp16_str = cls._require_env('BGE_RERANKER_FP16')
            fp16 = fp16_str.lower() in ("true", "1")

            # 2. 创建
            reranker = FlagReranker(
                model_name_or_path=model_name_or_path,
                device=device,
                use_fp16=fp16
            )

            return reranker
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE-M3重排序模型客户端创建失败: {e}") from e

if __name__ == '__main__':
    llm_client: ChatOpenAI = AIClients.get_llm_client()

    llm_response = llm_client.invoke("请您给我讲一个笑话，要求输出格式是一个json")

    llm_result = llm_response.content

    import json

    result = json.loads(llm_result)

    print(result)
