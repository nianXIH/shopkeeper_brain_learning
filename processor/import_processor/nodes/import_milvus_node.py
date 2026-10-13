from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, MilvusError
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class _SCALAR_FIELD_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FIELDS: [_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusSchemaBuilder():
    """
    负责处理和Milvus字段约束相关的逻辑
    """

    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        """
        创建schema
        Args:
            milvus_client:
            dim:

        Returns:
        enable_dynamic_field:动态字段：添加数据的时候可以提供在静态字段基础上额外的字段---静态字段（提前定义的）：

        """
        # 1. 创建schema
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 2. 添加字段约束
        # 2.1 添加主键字段的约束 auto_id:1.能够自动生成值  2. 在一定的时间内有顺序 3.插入记录的时候无需提供该字段
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

        # 2.2 添加向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2.3 添加标量字段的约束[标量字段个数：5个]
        for spec in _SCALAR_FIELDS:
            kwargs: Dict = {
                "field_name": spec.field_name,
                "datatype": spec.datatype
            }
            if spec.max_length:
                kwargs['max_length'] = spec.max_length

            schema.add_field(**kwargs)

        return schema


class _MilvusInserter:

    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert_rows(self, data: List[Dict[str, Any]]):
        # 1. 插入
        inserted_result = self._milvus_client.insert(collection_name=self._collection_name, data=data)

        # 2. 得到每一个chunk的id
        chunk_ids = inserted_result.get('ids')

        # 3. 回填到chunk中
        for chunk_id, chunk in zip(chunk_ids, data):
            chunk['chunk_id'] = chunk_id


class _MilvusIndexBuilder:

    @staticmethod
    def build_index_params(milvus_client: MilvusClient):
        index = milvus_client.prepare_index_params()

        # 稠密向量：AUTOINDEX
        index.add_index(field_name="dense_vector",
                        index_name="dense_vector_index",
                        index_type="AUTOINDEX",
                        metric_type="COSINE")

        # 稀疏向量：倒排索引
        index.add_index(field_name="sparse_vector",
                        index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP")

        return index


class ImportMilvusNode(BaseNode):
    """
    角色：充当门面（门面模式）
    """
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:

        Returns:

        """

        # 1. 校验state
        validated_chunks, dim = self._validate_state(state)

        # 2. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"MilVus客户端创建失败,异常原因{str(e)}")
            raise MilvusError(message=f"MilVus客户端创建失败,异常原因{str(e)}", node_name=self.name)

        # 3. 获取chunks集合
        chunks_collection = self.config.chunks_collection

        # 4. 创建集合
        self._create_chunks_collection(chunks_collection, milvus_client, dim)

        # 5. 插入数据(静态字段都要提供)
        _inserter = _MilvusInserter(milvus_client, chunks_collection)

        _inserter.insert_rows(validated_chunks)

        # 6. 返回state
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:

        self.log_step("validate", "参数校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError("待入库的 chunks 为空或类型无效", self.name)

        validated_chunks = []
        for i, chunk in enumerate(chunks):
            # 类型不对 → 抛异常（和上游 embedding 节点保持一致）
            if not isinstance(chunk, dict):
                raise ValidationError(
                    f"chunks[{i}] 类型无效：期望 dict，实际为 {type(chunk).__name__}", self.name
                )
            # 缺少向量 → 跳过（嵌入可能部分失败，属于数据级容错）
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                # item_name 在 Milvus schema 中为非空且无默认值的 VARCHAR，
                # 上游若未回填则用 file_title 兜底，避免整批插入失败。
                if not chunk.get("item_name"):
                    fallback = chunk.get("file_title") or ""
                    self.logger.warning(
                        f"chunks[{i}] 缺失 item_name，已用 file_title 兜底：{fallback!r}"
                    )
                    chunk["item_name"] = fallback
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")

        if not validated_chunks:
            raise ValidationError("所有 chunk 均无有效向量，无法入库", self.name)

        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效 chunks：{len(validated_chunks)}，向量维度：{dim}")
        return validated_chunks, dim

    def _create_chunks_collection(self, chunks_collection: str, milvus_client: MilvusClient, dim: int):

        # 1. 判断集合
        if milvus_client.has_collection(chunks_collection):
            self.logger.info(f"{chunks_collection}已存在 跳过创建")
            return
        # 2. 创建schema
        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)

        # 3. 创建索引
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)

        # 4. 创建集合
        milvus_client.create_collection(collection_name=chunks_collection, schema=schema, index_params=index_params)


def _cli_main() -> None:
    import json
    from pathlib import Path
    setup_logging()

    temp_dir = Path(
        r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp"
        )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get('chunks')}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    _cli_main()
