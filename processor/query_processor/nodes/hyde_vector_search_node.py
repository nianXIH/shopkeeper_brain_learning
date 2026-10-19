from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_processor.nodes.hybrid_search_base import HybridSearchBaseNode
from knowledge.utils.client.ai_clients import AIClients
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE


class HyDeVectorSearchNode(HybridSearchBaseNode):
    """
    HyDE 检索：先用LLM生成假设性文档（缓解query与文档的跨域不对称），
    再用「问题 + 假设性文档」做混合向量检索。
    公共流程（校验/客户端获取/嵌入/检索）全部继承自 HybridSearchBaseNode
    """
    name = "hyde_vector_search_node"
    result_key = "hyde_embedding_chunks"

    def _build_search_texts(self, rewritten_query: str, item_names: List[str]) -> Optional[List[str]]:
        # 1. 先用LLM生成原始问题对应的假设性文档（None表示生成失败，触发基类降级）
        hy_document = self._generate_hy_document(rewritten_query, item_names)
        if hy_document is None:
            return None

        # 2. 问题 + 假设性文档一起送去嵌入
        return [f"{rewritten_query}\n{hy_document}"]

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


if __name__ == '__main__':

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
