from typing import Tuple, List, Dict, Any
import math
from FlagEmbedding import FlagReranker
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):
    name = "reranker_node"

    @staticmethod
    def _sigmoid(score: float) -> float:
        """sigmoid归一化，将 (-∞, +∞) 映射到 (0, 1)"""
        return 1.0 / (1.0 + math.exp(-score))


    def process(self, state: QueryGraphState) -> QueryGraphState:
        """

        Args:
            state:

        Returns:

        """

        # 1. 获取用户问题
        user_query = state.get("rewritten_query") or state.get("original_query")

        # 2. 获取两路检索结果(本地检索结果,远程检索结果)
        rerank_outputs: List[Dict[str,Any]] = self._collect_rerank_inputs(state)

        # 3. 利用Reranker进行精排
        refine_docs: List[Dict[str, Any]] = self._refine_rank(user_query, rerank_outputs)

        # 4.top_k: 静态top_k 设置过大或者过小都有问题。能否使用动态top_k
        # 动态top_k:不是初始的时候不给top_k,初始的时候该给还是给
        # min_top_k(3):不管如何裁减，至少都要留下三个
        # max_top_k(10):最多留下10个
        # 不过这并不是最终的（可能会调整）【尽量减少幻觉，完全解决幻觉】
        #  top_k:如果设置大了 1.LLM上下文过长    2. 可能截取无关上下文(不该检索的确检索到了)--->LLM---> (幻觉)
        #  top_k:如果设置小了 1.LLM上下不会过程  2. 检索到上下文信息不全面(该检索的没有检索到)--->LLM--->(幻觉)

        reranked_docs = self._cliff_cutoff(
            refine_docs,
            self.config.rerank_min_top_k,
            self.config.rerank_max_top_k,
            self.config.rerank_gap_abs,
            self.config.rerank_gap_ratio,
        )

        state['reranked_docs'] = reranked_docs

        return state



    def _collect_rerank_inputs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """
        获取两路检索结果(本地检索结果、远程检索结果)
        Args:
            state:

        Returns:

        """
        final_docs = []
        # 1. 获取本地检索结果
        rrf_chunks = state.get('rrf_chunks') or []
        for chunk in rrf_chunks:
            # 1. 判断chunk
            if not chunk or not isinstance(chunk, dict):
                continue

            # 2. 获取chunk的信息
            # 2.1 获取chunk中content
            content = chunk.get('content', '')
            if not content:
                continue
            # 2.2 获取chunk中的title
            title = chunk.get('title', '')

            # 2.3 获取chunk中的chunk_id(一定有)
            chunk_id = chunk.get('chunk_id')

            # 3. 格式化文档(格式化本地)
            formated_local_doc = self._format_doc(content=content, chunk_id=chunk_id, title=title, source="local")

            final_docs.append(formated_local_doc)

        # 2. 获取远程检索结果
        web_search_docs = state.get('web_search_docs') or []
        for doc in web_search_docs:

            # 1. 判断doc
            if not doc or not isinstance(doc, dict):
                continue

            # 2. 获取content
            content = doc.get('snippet', '')
            # 2.3 获取title
            title = doc.get('title', '')
            # 2.4 获取url
            url = doc.get('url', '')

            # 3. 格式化文档(格式化远程)
            formated_web_doc = self._format_doc(content=content, title=title, url=url, source="web")
            final_docs.append(formated_web_doc)

        self.logger.info(f"获取Reranker阶段需要的搜索结果个数{len(final_docs)}")
        return final_docs


    def _format_doc(self, content: str, chunk_id: int = None, title: str = "", url: str = "", source: str = ""):
        """
        格式化本地以及远程检索到的文档
        Args:
            content:
            chunk_id:
            title:
            source:
            url:

        Returns:

        """
        return {
            "content": content,
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
            "source": source
        }

    def _refine_rank(self, user_query: str, rerank_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        reranker模型进行打分&排序【精排】
        Args:
            user_query:  用户的查询
            rerank_outputs: 本地和远程融合后的检索结果

        Returns:
            Dict[str,Any]:{"score","","other":"..."}

        """
        if not rerank_outputs:
            return []

        # 1. 获取重排序模型
        try:
            rerank_client: FlagReranker = AIClients.get_bge_m3_rerank_client()
        except ConnectionError as e:
            self.logger.error(f"获取BGE-M3重排序模型失败 原因:{str(e)}")
            return [{**d, "score": None} for d in rerank_outputs]

        # 2. 构建Q->D的pair对
        query_doc_pairs = [(user_query, d.get('content')) for d in rerank_outputs]

        try:
            # 3. 计算(注意：BGE-M3重排序模型计算出来的得分可以很大也可以很小(负无穷大,正无穷大))
            rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_pairs)

            # 4.组合最终结果
            doc_score = [{**d, "score": self._sigmoid(float(s))} for d, s in zip(rerank_outputs, rerank_scores)]

            # 5. 排序
            sorted_doc_score = sorted(doc_score, key=lambda x: x['score'], reverse=True)

            # 6. 返回
            return sorted_doc_score
        except Exception as e:
            self.logger.error(f"BGE-M3重排序模型计算分数失败 原因：{str(e)}")
            return [{**d, "score": None} for d in rerank_outputs]

    def _cliff_cutoff(self, refine_docs: List[Dict[str, Any]], rerank_min_top_k: int, rerank_max_top_k: int,
                      rerank_gap_abs: float, rerank_gap_ratio: float) -> List[Dict[str, Any]]:
        """
          动态top_k: 双重阈值裁剪(绝对阈值 + 相对阈值)
          1. 绝对阈值(rerank_gap_abs): 分数低于该值的文档相关性不足, 直接裁掉
          2. 相对阈值(rerank_gap_ratio): 相邻文档分数相对降幅超过该比例视为断崖, 从断崖处截断
          两条裁剪线取更严者, 再用 min_top_k 兜底
        """

        upper_bound = min(rerank_max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)
        cut_off = upper_bound
        max_gap = 0

        # 1. 绝对阈值裁剪: 分数已降序, 从头找第一个分数低于绝对阈值的文档
        for i, doc in enumerate(refine_docs[:upper_bound]):
            score = doc.get('score')
            if score is not None and score < rerank_gap_abs:
                cut_off = min(cut_off, i)
                self.logger.info(f"位置{i}分数{score:.4f}低于绝对阈值{rerank_gap_abs}发生裁剪")
                break

        # 2. 相对阈值裁剪: 找相对降幅达标的全局最大断崖
        for i in range(0, upper_bound - 1):
            current_score = refine_docs[i].get('score')
            next_score = refine_docs[i + 1].get('score')

            if current_score is None or next_score is None or current_score <= 0:
                continue

            gap = current_score - next_score
            ratio = gap / current_score

            if ratio >= rerank_gap_ratio and gap > max_gap:
                max_gap = gap
                cut_off = min(cut_off, i + 1)
                self.logger.info(f"位置{i + 1}发生相对断崖(相对降幅{ratio:.2%})")

        # 兜底：不管如何裁剪，至少保留 lower_bound 个
        cut_off = max(cut_off, lower_bound)

        cutoff_docs = refine_docs[:cut_off]

        return cutoff_docs


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)