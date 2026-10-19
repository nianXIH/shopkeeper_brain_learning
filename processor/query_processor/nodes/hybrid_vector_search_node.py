from typing import List, Optional

from knowledge.processor.query_processor.nodes.hybrid_search_base import HybridSearchBaseNode


class HybridVectorSearch(HybridSearchBaseNode):
    """
    直接用「LLM重写后的问题」做混合向量检索（dense + sparse）
    公共流程（校验/客户端获取/嵌入/检索）全部继承自 HybridSearchBaseNode
    """
    name = "hybrid_vector_search_node"
    result_key = "embedding_chunks"

    def _build_search_texts(self, rewritten_query: str, item_names: List[str]) -> Optional[List[str]]:
        # 直接检索：嵌入重写后的用户问题
        return [rewritten_query]


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
