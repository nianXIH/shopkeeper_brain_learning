import threading
from typing import Optional
from typing import TypeVar, Optional
import logging

logger = logging.getLogger(__name__)

import urllib3
from urllib3.util import Timeout
from minio import Minio
from pymilvus import MilvusClient
from dotenv import load_dotenv
from knowledge.utils.client.base import BaseClientManager

load_dotenv()

# MinIO HTTP 客户端的超时与重试策略
# connect: 建立 TCP 连接的超时(秒)，避免 endpoint 不通时无限挂起
# read:    单次请求响应读取的超时(秒)，覆盖大文件上传场景
_MINIO_CONNECT_TIMEOUT = 5
_MINIO_READ_TIMEOUT = 60


def _build_minio_http_client() -> urllib3.PoolManager:
    """构建带超时/重试的 urllib3 客户端，注入给 Minio SDK。"""
    return urllib3.PoolManager(
        timeout=Timeout(connect=_MINIO_CONNECT_TIMEOUT, read=_MINIO_READ_TIMEOUT),
        retries=urllib3.Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "PUT", "HEAD", "DELETE"]),
        ),
    )


class StorageClients(BaseClientManager):
    """存储类客户端：MinIO、Milvus"""

    _minio_client: Optional[Minio] = None
    _minio_lock = threading.Lock()

    _milvus_client: Optional[MilvusClient] = None
    _milvus_lock = threading.Lock()

    # ── MinIO ──

    @classmethod
    def get_minio_client(cls) -> Minio:
        return cls._get_or_create("_minio_client", cls._minio_lock, cls._create_minio_client)

    @classmethod
    def _create_minio_client(cls) -> Minio:
        try:
            endpoint = cls._require_env("MINIO_ENDPOINT")
            access_key = cls._require_env("MINIO_ACCESS_KEY")
            secret_key = cls._require_env("MINIO_SECRET_KEY")
            bucket_name = cls._require_env("MINIO_BUCKET_NAME")

            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False,
                http_client=_build_minio_http_client(),
            )

            # 显式捕获 S3 层错误，把服务端返回的 code(如 RequestTimeTooSkewed / SignatureDoesNotMatch)
            # 打到日志里，避免上层拿到 ConnectionError 却看不到真正原因
            try:
                exists = client.bucket_exists(bucket_name)
                if not exists:
                    client.make_bucket(bucket_name)
                    logger.info(f"MinIO bucket '{bucket_name}' 已自动创建")
                else:
                    logger.info(f"MinIO bucket '{bucket_name}' 已存在")
            except Exception as bucket_err:
                logger.error(
                    f"MinIO bucket '{bucket_name}' 检查/创建失败 endpoint={endpoint}: {bucket_err}"
                )
                raise

            logger.info(f"MinIO 客户端初始化成功 (endpoint={endpoint})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            raise ConnectionError(f"MinIO 连接失败: {e}") from e

    # ── Milvus ──
    @classmethod
    def get_milvus_client(cls) -> MilvusClient:
        return cls._get_or_create("_milvus_client", cls._milvus_lock, cls._create_milvus_client)

    @classmethod
    def _create_milvus_client(cls) -> MilvusClient:
        try:

            milvus_uri = cls._require_env('MILVUS_URL')

            milvus_client = MilvusClient(uri=milvus_uri)

            return milvus_client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"Milvus 客户端创建失败: {e}")
            raise ConnectionError(f"Milvus 连接失败: {e}") from e


if __name__ == '__main__':
    print(StorageClients.get_milvus_client())
