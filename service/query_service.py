import uuid, logging

# 日志配置只应在应用入口（router的create_app）统一做，库模块导入即basicConfig属于反模式
logger = logging.getLogger(__name__)

from typing import List,Dict,Any
from knowledge.processor.query_processor.main_graph import query_app
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED
from knowledge.utils.task_util import get_task_result
from knowledge.utils.mongo_history_util import get_recent_messages
from knowledge.utils.mongo_history_util import clear_history
from knowledge.utils.sse_util import push_sse_event, SSEEvent


class QueryService:
    """

    """

    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id():
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, session_id: str, task_id: str, query: str, is_stream: bool):
        """
        运行查询流程的pineline
        Args:
            session_id:  会话id
            task_id:     任务id
            query:       查询问题
            is_stream:   是否是流式

        Returns:

        """
        # 1. 修改任务状态为正在执行
        update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)

        # 2. 构建查询初始化状态
        query_init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "is_stream": is_stream
        }

        # 3. 执行
        try:
            # 3.1 执行查询流程的pineline(调用的是CompiledStateGraph的invoke())
            query_app.invoke(query_init_state)
            # 3.2 更新整个任务状态为完成状态
            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)

        # 4. 更新整个任务状态为异常状态
        except Exception as e:
            # 查询链路异常是关键故障：exception会连完整堆栈一起落盘，error(str(e))会丢堆栈无法定位
            logger.exception(f"运行查询流程出现异常:{str(e)}")
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)
            # 流式任务失败也必须推送FINAL事件：否则SSE通道永远不关闭，
            # 前端会一直停在"进行中"转圈，无法感知任务已经失败
            if is_stream:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL,
                               data={"answer": f"抱歉，本次查询处理失败，请稍后重试。\n失败原因:{str(e)}",
                                     "error": str(e)})

    def get_task_result(self, task_id: str):
        answer = get_task_result(task_id=task_id, key="answer")
        return answer

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:

        # 1. 根据session_id获取最近的指定条数的历史对话
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        return clear_history(session_id)
