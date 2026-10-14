import os.path
import logging
import shutil
import time
import uuid
from datetime import datetime
from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, add_node_duration, \
    TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UpLoadService:
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
            logger.error(f"[{task_id}] 执行导入过程中出现异常 原因{str(e)}")

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

        # 5. 保存文件到minio中
        self.save_upload_file_to_minio(import_file_path, file.filename)
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

        # 2. 构建导入文件的路径
        import_file_path = os.path.join(file_dir, file.filename)

        # 3. 写入
        try:
            with  open(import_file_path, "wb") as f:
                # shutil.copyfileobj() 不同的操作系统以及不同python版本都可以分批次的写入（windows版本以及3.7以上的sdk版本:1m）
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.info(f"{file.filename}写入临时目录失败 原因:{str(e)}")
            raise FileProcessingError(message=f"{file.filename}写入临时目录失败 原因:{str(e)}")

        # 4. 返回导入的文件路径
        return import_file_path

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
        object_name = f"origin_files/{datetime.now().strftime('%Y%m%d')}/{filename}"

        # 3. 上传
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f"{filename}上传到MinIO失败 原因：{str(e)}")
