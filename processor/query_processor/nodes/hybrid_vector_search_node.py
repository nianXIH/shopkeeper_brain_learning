from typing import Tuple, List, Dict, Any,Union
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, _item_names_filter


class HybridVectorSearch(BaseNode):
    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:
        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 2. 获取嵌入模型客户端
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return state

        # 3. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return state

        # 4. 嵌入、检索
        # 4. 嵌入、检索
        try:
            # 4.1 嵌入查询问题
            embed_query_vector = generate_bge_m3_hybrid_vectors(model=bge_m3_client,
                                                                embedding_documents=[rewritten_query])
        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}嵌入获取失败 原因:{str(e)}")
            return state


        try:
            # 4.2 创建混合检索请求(expr:对检索的返回做过滤的)
            expr, expr_params = _item_names_filter(item_names)
            hybrid_search_req = create_hybrid_search_requests(dense_vector=embed_query_vector['dense'][0],
                                                              sparse_vector=embed_query_vector['sparse'][0],
                                                              expr=expr,
                                                              expr_params=expr_params,
                                                              limit=5
                                                              )
            # 4.3 执行混合搜索请求
            hybrid_search_res = execute_hybrid_search_query(milvus_client=milvus_client,
                                                            collection_name=self.config.chunks_collection,
                                                            search_requests=hybrid_search_req,
                                                            ranker_weights=(0.5, 0.5),
                                                            norm_score=True,
                                                            limit=5,
                                                            output_fields=["chunk_id", "content", "item_name", 'title']
                                                            )

            if not hybrid_search_res or not hybrid_search_res[0]:
                self.logger.warning(f"未检索到相关文档")
                return state

            # 4.4 更新state
            # 4.5  更新自己修改的内容,返回自己修改的.
            return {"embedding_chunks": hybrid_search_res[0]}

        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}执行混合搜索查询失败 原因:{str(e)}")
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

if __name__ == '__main__':
    import json

    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]
    }

    vector_search = HybridVectorSearch()
    result = vector_search.process(state)

    embedding_chunks = result.get('embedding_chunks')
    if not embedding_chunks:
        print("未检索到相关文档")
    else:
        for r in embedding_chunks:
            print(json.dumps(r, ensure_ascii=False, indent=2))
