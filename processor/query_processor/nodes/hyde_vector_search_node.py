from typing import Tuple, List, Dict, Any, Optional, Union

from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, _item_names_filter


class HyDeVectorSearchNode(BaseNode):
    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:

        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 2. 利用LLM生成原始问题对应的假设性答案（解决跨域不对称）
        hy_document = self._generate_hy_document(rewritten_query, item_names)

        # 3. 判断
        if hy_document is None:
            return state

        # 4. 获取嵌入模型以及milvus客户端
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return state

        # 5. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return state

        # 6. 为假设性文档嵌入
        try:
            embed_hy_vector = generate_bge_m3_hybrid_vectors(bge_m3_client, [f"{rewritten_query}\n{hy_document}"])
        except Exception as e:
            self.logger.error(f"假设性文档{hy_document}嵌入获取失败 原因:{str(e)}")
            return state

        # 7. 向量检索
        try:
            # 7.1 创建混合检索请求(expr:对检索的返回做过滤的)
            expr, expr_params = _item_names_filter(item_names)
            hybrid_search_req = create_hybrid_search_requests(dense_vector=embed_hy_vector['dense'][0],
                                                              sparse_vector=embed_hy_vector['sparse'][0],
                                                              expr=expr,
                                                              expr_params=expr_params,
                                                              limit=5
                                                              )
            # 7.2 执行混合搜索请求
            hybrid_search_res = execute_hybrid_search_query(milvus_client=milvus_client,
                                                            collection_name=self.config.chunks_collection,
                                                            search_requests=hybrid_search_req,
                                                            ranker_weights=(0.5, 0.5),
                                                            norm_score=True,
                                                            limit=5,
                                                            output_fields=["chunk_id", "content", "item_name", 'title']
                                                            )

            if not hybrid_search_res or not hybrid_search_res[0]:
                return state


            # 7.3 修改自己的并且返回修改后的
            return {"hyde_embedding_chunks":hybrid_search_res[0]}

        except Exception as e:
            self.logger.error(
                f"原始问题{rewritten_query}对应的假设性文档{hy_document}执行混合搜索查询失败 原因:{str(e)}")
            return state

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        # 1. 用户的问题（LLM重写后的）
        rewritten_query = state.get('rewritten_query')

        # 2. 获取商品名列表
        item_names = state.get('item_names')

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names

    def _generate_hy_document(self, rewritten_query: str, item_names: List[str]) -> Optional[str]:
        """
        生成假设性文档
        Args:
            rewritten_query:
            item_names:


        Returns:

        """

        # 1. 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"获取LLM客户端失败 原因:{str(e)}")
            return None

        # 2. 获取提示词
        # 2.1 获取系统提示词
        system_prompt = f"您是一位{item_names}方面的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        # 2.2 获取用户提示词
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(item_names=item_names, rewritten_query=rewritten_query)

        # 3. 调用
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            self.logger.error(f"LLM生成{item_names}的假设性文档失败 原因:{str(e)}")
            return None

        # 4. 判断是否有内容
        if not llm_response.content.strip():
            return None

        # 5. 返回
        return llm_response.content.strip()


if __name__ == "__main__":

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    node = HyDeVectorSearchNode()
    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n【HyDE 检索结果】: {len(chunks)} 条")
    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"  [{i}] chunk_id={entity.get('chunk_id')} "
              f"item_name={entity.get('item_name')} "
              f"distance={chunk.get('distance', 'N/A')}")
        content = entity.get("content", "")
        print(f"      内容: {content[:80]}...")
