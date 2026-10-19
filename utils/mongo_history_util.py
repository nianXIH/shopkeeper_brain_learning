import logging
import threading
import time
from typing import List, Dict, Any
from bson import ObjectId
from pymongo.collection import Collection
from pymongo import DESCENDING

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "chat_message"

# 索引只需建一次：避免每次获取集合都触发 create_index 往返
_ensure_index_done = False
_ensure_index_lock = threading.Lock()


def _ensure_history_index(collection: Collection) -> None:
    """确保 (session_id, ts) 复合索引存在（幂等）。

    get_recent_messages 的查询模式是 session_id 等值 + ts 排序 + limit，
    无索引时每次拉历史都是全表扫描 + 内存排序，会话多了之后是慢查询源头。
    create_index 在索引已存在时是 no-op，进程内只真正执行一次。
    """
    global _ensure_index_done
    if _ensure_index_done:
        return
    with _ensure_index_lock:
        if _ensure_index_done:
            return
        try:
            # ts 用 DESCENDING 与 get_recent_messages 的排序方向一致，排序可直接走索引
            collection.create_index([("session_id", 1), ("ts", -1)])
            _ensure_index_done = True
        except Exception as e:
            # 索引创建失败不应阻断历史读写主流程，仅记录告警
            logger.warning(f"创建chat_message复合索引失败: {e}")


def _get_collection() -> Collection:
    """获取 chat_message 集合"""
    collection = StorageClients.get_mongo_db()[_COLLECTION_NAME]  # "chat_message"表名
    _ensure_history_index(collection)
    return collection


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        message_id: str = None,
) -> str:
    """
    MongoDB的写入操作
    新增(message_id如果为空) or  修改（message_id不为空）
    Args:
        session_id:
        role:
        text:
        rewritten_query:
        item_names:
        message_id:

    Returns:

    """
    # 直接取系统时间戳：datetime.now()受本地时区影响，
    # 多机部署/时区漂移时会打乱ts排序，导致历史对话顺序错乱
    ts = time.time()

    # 1. 构建记录结构
    document = {
        "session_id": session_id,  # 会话id
        "role": role,  # 角色
        "text": text,  # 内容
        "rewritten_query": rewritten_query,  # 重写后问题
        "item_names": item_names or [],  # 商品名列表
        "ts": ts,  # 时间戳
    }

    # 2. 获取集合[客户端 db collection]
    collection = _get_collection()
    if message_id:
        collection.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = collection.insert_one(document)
        return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        # 按时间倒序取出最近 N 条后再反转为时间正序：
        # 调用方拿到的历史必须是"从旧到新"的时间线，直接拼给 LLM 顺序才正确
        return list(cursor)[::-1]
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []


def clear_history(session_id: str) -> int:
    try:
        result = _get_collection().delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0
