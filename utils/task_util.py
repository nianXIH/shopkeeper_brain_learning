from typing import Any, Dict, List
from collections import defaultdict


"""
任务id: 主要追踪上传文件（任务）的状态流程的
上传一个文件就属于一个任务（唯一的任务id）---未来如果想知道上传的这个文件处理到哪里了就需要用任务id来查询
"""
# 只要访问不存在的 key，自动帮你初始化为 []
_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)

# 只要访问不存在的 key，自动帮你初始化为 {} 查询时候用
# key：任务id    内部字典的key:任务对应的某一个节点（答案生成节点）--->节点对应的结果
# {"task1":{"answer":"节点的结果"}}
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)

_tasks_status: Dict[str, str] = {}

TASK_STATUS_PROCESSING = "processing"  # 任务处理中
TASK_STATUS_COMPLETED = "completed"  # 任务完成
TASK_STATUS_FAILED = "failed"  # 任务失败

_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "上传文件",
    "entry_node": "检查文件",
    "pdf_to_md_node": "PDF转Markdown",
    "md_to_img_node": "Markdown图片处理",
    "document_split_node": "文档切分",
    "item_name_recognition_node": "主体名称识别",
    "embedding_chunk_node": "向量生成",
    "import_milvus_node": "导入向量数据库",
    "__end__": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "item_name_confirmed_node": "确认问题产品",
    "answer_output_node": "生成答案",
    "reranker_node": "重排序",
    "rrf_merge_node": "倒排融合",
    "web_mcp_search_node": "网络搜索",
    "hybrid_vector_search_node": "切片搜索",
    "hyde_vector_search_node": "切片搜索(假设性文档)"
}


def _to_cn(node_name: str) -> str:
    # 1. 从节点映射字典中获取中文名，若未配置则直接返回原英文名
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    # 1. 获取当前任务的运行节点列表（利用 defaultdict 自动初始化特性）
    running = _tasks_running_list[task_id]

    # 2. 将当前节点加入运行列表（并做去重判断，防止重复添加）
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    # 1. 如果该节点还在运行列表中，则将其移出（表示该节点已结束运行）
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)

    # 2. 获取当前任务的已完成节点列表
    done = _tasks_done_list[task_id]

    # 3. 将当前节点加入已完成列表（做去重判断，防止重复标记）
    if node_name not in done:
        done.append(node_name)


def get_running_task_list(task_id: str) -> List[str]:
    # 1. 获取指定任务运行中的节点列表，并通过列表推导式统一转换为中文展示名返回
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def get_done_task_list(task_id: str) -> List[str]:
    # 1. 获取指定任务已完成的节点列表，并通过列表推导式统一转换为中文展示名返回
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
    """
    根据任务ID 获取任务状态
    :param task_id:
    :return:
    """
    # 1. 安全获取指定任务的总体运行状态，若不存在则返回空字符串
    return _tasks_status.get(task_id, "")




def update_task_status(task_id: str, status_name: str) -> None:
    # 1. 更新指定任务的总体运行状态（如 processing 等）
    _tasks_status[task_id] = status_name


def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    获取任务结果字段（如 answer / error）。
    """
    return _tasks_result.get(task_id, {}).get(key, default)



def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    """记录节点耗时（秒）"""
    cn_name = _to_cn(node_name)
    _tasks_duration[task_id][cn_name] = round(duration, 2)

def get_node_durations(task_id: str) -> Dict[str, float]:
    """获取所有节点的耗时"""
    return dict(_tasks_duration.get(task_id, {}))

def get_task_info(task_id: str) -> Dict[str, Any]:
    """
    获取任务的全局信息（状态 + 运行中节点 + 已完成节点）
    :param task_id: 任务ID
    :return: 包含 status、running_list、done_list 的字典
    """
    return {
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
        "durations": get_node_durations(task_id)
    }

