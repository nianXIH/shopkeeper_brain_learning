import logging
from typing import List, Dict, Any
from datetime import datetime
from bson import ObjectId
from pymongo.collection import Collection
from pymongo import DESCENDING

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection() -> Collection:
    """获取 chat_message 集合"""
    return StorageClients.get_mongo_db()["chat_message"]  # "chat_message"表名


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
    ts = datetime.now().timestamp()

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
        return list(cursor)
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
