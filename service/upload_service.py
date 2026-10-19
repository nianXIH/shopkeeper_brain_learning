import os.path
import logging
import time
import uuid
from datetime import datetime
from typing import Optional
from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, add_node_duration, \
    TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED

# 日志配置只应在应用入口（router的create_app）统一做，库模块导入即basicConfig属于反模式
logger = logging.getLogger(__name__)


class UpLoadService:
    # 上传文件约束：后缀白名单与大小上限（路径穿越/非法类型/超大文件都必须在落盘前拦截）
    _ALLOWED_SUFFIXES = {".pdf"}
    _MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    def get_base_dir(self) -> str:

        return os.path.join(get_local_base_dir(),datetime.now().strftime("%Y%m%d"))

    """
    处理文件上传的相关逻辑
    """

    def run_import_graph(self, task_id: str, import_file_path: str, file_dir: str):
        """
        运行整个图谱流程
        Args:
            task_id:
            import_file_path:
            file_dir:

        Returns:

        """

        # 1. 更新任务状态为processing
        update_task_status(task_id, TASK_STATUS_PROCESSING)

        # 2. 定义运行graph流程的状态
        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }


        # stream:迭代整个graph图状态可以得到每一个节点的事件(节点的名字以及节点操作完state之后的新状态)
        # 3. 运行整个导入图状态
        try:
            for event in import_app.stream(graph_state):

                for key, value in event.items():
                    logger.info(f"当前正在执行的节点--->{key}")

            # 3.1 更新任务为已完成
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            # 导入链路异常是关键故障：exception会连完整堆栈一起落盘，error(str(e))会丢堆栈无法定位
            logger.exception(f"[{task_id}] 执行导入过程中出现异常 原因{str(e)}")

            # 3.2 更新任务为失败
            update_task_status(task_id, TASK_STATUS_FAILED)

    def process_upload_file(self, file: UploadFile):
        """
        处理文件上传

        1. 将上传的文件存储到本地临时目录(主要为了做中转)
        2. 将上传的文件存储到远程minio(主要持久化)
        3. 将file_dir / import_file_path /task_id 返回
        Args:
            file:

        Returns:

        """

        # 1. 生成任务id

        task_id = str(uuid.uuid4().hex[:8])  # 真正的随机 获取前8个随机数
        add_running_task(task_id, "upload_file")
        start_time = time.time()

        # 2. 生成日期目录并且将日期目录和临时目录拼接到一起
        base_file_dir = self.get_base_dir()

        # 3. 构建文档完整归属目录
        file_dir = os.path.join(base_file_dir, task_id)

        # 4. 保存文件到临时目录
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        # 5. 保存文件到minio中（对象名使用清洗后的安全文件名，不信任原始filename）
        self.save_upload_file_to_minio(import_file_path, os.path.basename(import_file_path))
        end_time = time.time()
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id, "upload_file", end_time - start_time)

        # 6. 返回图谱的信息
        return task_id, import_file_path, file_dir

    def save_upload_file_to_local(self, file: UploadFile, file_dir: str) -> str:
        """
        保存文件到临时目录
        Args:
            file: 文件上传对象
            file_dir: 上传文件的目录

        Returns:

        """
        # 1. 创建文件的归属目录
        os.makedirs(file_dir, exist_ok=True)

        # 2. 清洗文件名：只保留最后一段路径，防止 ../ 路径穿越写到目录外
        safe_filename = self._sanitize_filename(file.filename)

        # 3. 后缀白名单校验（在落盘前拦截，而不是等后台任务的entry_node才失败）
        if os.path.splitext(safe_filename)[1].lower() not in self._ALLOWED_SUFFIXES:
            raise FileProcessingError(
                message=f"不支持的文件类型:{safe_filename}，仅支持{sorted(self._ALLOWED_SUFFIXES)}")

        # 4. 构建导入文件的路径
        import_file_path = os.path.join(file_dir, safe_filename)

        # 5. 分块写入并限制总大小（防止超大文件写满磁盘）
        total_size = 0
        try:
            with open(import_file_path, "wb") as f:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > self._MAX_UPLOAD_SIZE:
                        raise FileProcessingError(
                            message=f"{safe_filename}超过大小上限{self._MAX_UPLOAD_SIZE // (1024 * 1024)}MB")
                    f.write(chunk)
        except IOError as e:
            logger.exception(f"{safe_filename}写入临时目录失败")
            raise FileProcessingError(message=f"{safe_filename}写入临时目录失败 原因:{str(e)}")

        # 6. 返回导入的文件路径
        return import_file_path

    def _sanitize_filename(self, filename: Optional[str]) -> str:
        """
        清洗上传文件名：统一路径分隔符后只取最后一段，杜绝 ../ 与 ..\\.. 两种路径穿越写法
        """
        name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or name in {".", ".."}:
            raise FileProcessingError(message="上传文件名无效")
        return name

    def save_upload_file_to_minio(self, import_file_path: str, filename: str):
        """

        Args:
            import_file_path:  上传文件的地址
            filename: 上传文件的名字

        Returns:

        """

        # 1. 获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"MinIO客户端获取失败 原因:{str(e)}")
            return

        # 2. 获取minio相关信息
        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        if not bucket_name:
            # os.getenv可能返回None：不拦截会让fput_object直接报错且难定位
            logger.error("缺少环境变量MINIO_BUCKET_NAME，跳过MinIO上传")
            return
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        # 3. 上传
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.exception(f"{filename}上传到MinIO失败")
