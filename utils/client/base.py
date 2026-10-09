import os
import logging
import threading
from typing import TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseClientManager:
    """
    客户端管理器基类，提供：
        - _require_env()：环境变量校验
        - _get_or_create()：双重检查锁模板方法

    子类只需要关注「怎么创建客户端」，不用重复写锁逻辑。
    """

    @staticmethod
    def _require_env(key: str) -> str:
        """读取必需的环境变量，缺失时立即抛异常。"""
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(f"缺少必需的环境变量: {key}")
        return value

    @classmethod
    def _get_or_create(cls, attr_name: str, lock: threading.Lock, factory):
        """
        双重检查锁的通用模板。

        Args:
            attr_name: 类属性名（如 "_minio_client"）
            lock: 对应的线程锁
            factory: 无参工厂函数，返回客户端实例

        Returns:
            缓存的或新创建的客户端实例
        """
        # 第一次检查（无锁，快速路径）
        instance = getattr(cls, attr_name, None)
        if instance is not None:
            return instance

        with lock:
            # 第二次检查（持锁，防并发重复创建）
            instance = getattr(cls, attr_name, None)
            if instance is not None:
                return instance

            instance = factory()
            setattr(cls, attr_name, instance)
            return instance
