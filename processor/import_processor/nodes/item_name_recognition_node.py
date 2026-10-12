from typing import List, Tuple, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.test.bge.embedding.test_bge_m3 import bge_m3_ef
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"

    def process(self,state:ImportGraphState)->ImportGraphState:
        """
        节点的处理逻辑入口
        1. 负责利用LLM提取商品的具体型号（名）
        2. 嵌入商品具体型号（名）
        3. 存储到Milvus中（MySQL：模糊查询的时候不会考虑语义）
        Args:
            state:

        Returns:
        """

        # 1. 参数校验
        file_title, chunks, item_name_chunk_k = self._validate_state(state)

        # 2. 构建上下文
        item_name_context = self._prepare_llm_context(chunks, item_name_chunk_k)

        # 3. 调用LLM模型 提取商品名
        item_name = self._recognition_item_name(item_name_context, file_title)

        # 4. 向量化(嵌入模型：1.OpenAIEmbedding(OpenAI) 2.文本嵌入模型（text-embedding-v(x))（灵积服务平台：dashscope） 3.bge(bge-m3))：混合向量[稠密：相似性匹配、稀疏：精确匹配]
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 5. 入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)

        # 6. 回填(更新LLM提取到的item_name)
        self._fill_item_name(state, chunks, item_name)

        return state

    def _validate_state(self, state) -> Tuple[str, List, int]:
        """

        Args:
            state:

        Returns:

        """

        # 1. 获取文档标题(商品具体型号[名]兜底)
        file_title = state.get("file_title")

        # 2. 判断文档标题
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name='file_title', expected_type=str)

        # 3. 获取chunks(供LLM作为上下文信息)
        chunks = state.get('chunks')

        # 4. 判断chunks
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name='chunks', expected_type=list)

        # 5. 获取item_name_chunk_k(3) 以及item_name_chunk_size(2500)
        item_name_chunk_k = self.config.item_name_chunk_k

        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message="商品名识别的辅助切片数不合法")

        # 6. 返回
        return file_title, chunks, item_name_chunk_k

    def _prepare_llm_context(self, chunks:List[Dict], item_name_chunk_k:int)->str:
        """
        职责：准备LLM的上下文信息
        Args:
            chunks: 切片列表
            item_name_chunk_k: 辅助切片数

        Returns:
            str: LLM的上下文信息
        """

        # 1. 遍历
        final_content = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):

            # 1.1 不是字典类型的直接跳过
            if not isinstance(chunk,dict):
                continue

            # 1.2 获取chunk的content
            content = chunk.get('content')

            splice_context = f"【切片】- {index}- {content}"

            final_content.append(splice_context)

        return "\n".join(final_content)

    def _recognition_item_name(self, item_name_context: str, file_title: str) -> str:

        # 1.调用LLM客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM客户端创建失败,降级使用文件标题{file_title}作为商品名 {str(e)}")
            return file_title

        # 2. 调用LLM模型
        # 2.1 获取商品名识别的llm系统提示词
        system_prompt = ITEM_NAME_SYSTEM_PROMPT

        # 2.2 获取商品名识别的llm用户提示词
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)

        try:
            # 3. 调用 返回AIMessage对象
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 4. 获取AI回复的具体内容
            llm_result = llm_response.content.strip('')
            if not llm_result or llm_result == 'UNKNOWN':
                self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名兜底")
                return file_title

            self.logger.info(f"LLM为文档：{file_title} 提取的商品名：{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名: {str(e)}")
            return file_title

    def _embedding_item_name(self,item_name:str)->Tuple[Optional[List],Optional[Dict[str,Any]]]:
        """
        商品名嵌入
        Args:
            item_name:商品名

        Returns:
            稠密向量,稀疏向量

        """
        try:
            bge_m3_client:BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE M3 客户端创建失败: {str(e)}")
            return None, None
        try:
            # 2. 计算稠密和稀疏向量
            vector_result = bge_m3_client.encode_documents(documents=[item_name])

            # 3. 解析稠密向量和稀疏向量
            # 3.1 获取稠密向量
            dense_vector = vector_result.get('dense')[0].tolist()
            # 3.2 获取稀疏向量矩阵csr(解开)
            sparse_csr = vector_result.get('sparse')
            # 3.1 获取行索引
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            # 3.2 获取token_id
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            # 3.3 获取weight
            weight = sparse_csr.data[start_index:end_index].tolist()
            # 3.4 构建{"token_id":weight}字典结构

            sparse_vector = dict(zip(token_id, weight))
            self.logger.info(f"计算出来的稠密向量的维度: {len(dense_vector)}")
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE-M3嵌入模型计算{item_name}向量失败 原因：{str(e)}")
            return None, None

    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any], file_title: str, item_name: str):
        """
        将LLM识别到的商品名保存到Milvus数据库中
        row行记录{"dense_vector":"值","sparse_vector":"值","file_title":"文档","item_name":"商品名"}
        Args:
            dense_vector: 商品名稠密向量
            sparse_vector: 商品名稀疏向量
            file_title: 文档名
            item_name:   商品名

        Returns:
        """

        # 1. 判断稠密向量和稀疏向量是否都存在
        if not dense_vector or not sparse_vector:
            return

        # 2. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败,原因：{str(e)}")
            return

        # 3. Milvus三大核心概念（集合：Collection[1.集合名 2.字段约束：schema 3.索引]）
        # 3.1 集合名：表名类似于归纳数据的容器，逻辑概念
        # 3.2 约束：类似于MySQL字段的长度、字段的类型..
        # 3.3 索引：类似于MySQL中的索引【索引类型比较多 B+树 Hash】。Milvus索引类型也有很多（专门针对于稠密向量的索引类型 针对于稀疏向量的索引类型 标量字段类型、主键类型）
        # 索引：本质就是算法（图、树、hash..）目的：提高检索【查询】效率。milvus中不管稠密向量索引还是稀疏向量的索引都是为了能够快速找到和问题相似的向量。
        # 使用Milvus的流程：①：创建集合（约束、索引）②：插入数据  ③：查询/检索

        # 4. 获取集合名字
        item_name_collection_name = self.config.item_name_collection
        try:
            # 5. 创建Milvus集合(幂等性校验)
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name,milvus_client)
            
            # 6. 构建索引
            item_name_data_row = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }

            # 7. 插入数据
            inserted_result = milvus_client.insert(collection_name=item_name_collection_name, data=[item_name_data_row])

            self.logger.info(f"插入的结果:{inserted_result},主键值:{inserted_result.get('ids')}")
        except Exception as e:
            self.logger.error(f"商品名{item_name}插入失败 {str(e)}")
            return

    def _create_item_name_collection(self, item_name_collection_name: str, milvus_client: MilvusClient):
        """
        创建商品名集合
        Args:
            item_name_collection_name: 集合的名字
            milvus_client: milvus客户端

        Returns:

        """
        # 1. 创建schema约束
        schema = milvus_client.create_schema()
        # 1.1 创建主键字段约束
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=10)

        # 1.2 创建标量字段的约束
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        # 1.3 创建向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引(标量字段建立索引 向量字段建立索引)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"  # IP COSINE L2
            # Milvus计算出来的稠密向量已经进行了归一化处理 所以度量类型选择COSINE或者IP效果一样，但是如果使用别的方式计算出来的稠密向量没有经过归一化处理 那么COSINE和IP就不相等
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"  # 只有IP 和BM25
        )

        # 3. 创建集合
        milvus_client.create_collection(collection_name=item_name_collection_name,
                                        schema=schema, index_params=index_params)

        self.logger.info(f"创建{item_name_collection_name}集合成功")

    def _fill_item_name(self, state: ImportGraphState, chunks: List[Dict], item_name: str):
        """
        回填item_name:
        位置一：回填给chunk 方便下游模型使用
        位置二：回填给state 方便其它节点使用
        Args:
            state:
            item_name:

        Returns:
        """
        # 1. 更新chunk的item_name
        for chunk in chunks:
            chunk['item_name'] = item_name

        # 2. 更新state中的item_name
        state['item_name'] = item_name

if __name__ == '__main__':
    import json

    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"C:\Users\a3384310973\Desktop\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\nodes\temp\chunks.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")
