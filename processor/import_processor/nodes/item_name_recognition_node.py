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
        file_title = state.get_field("file_title")

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






if __name__ == '__main__':
    import json

    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"D:\develop\develop\workspace\pycharm\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\chunks.json"
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
