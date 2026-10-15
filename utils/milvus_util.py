import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

from typing import Optional, List, Tuple, Any, Dict
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

milvus_client: Optional[MilvusClient] = None


# ------------------------------------------------------------------
# 创建混合检索请求
# ------------------------------------------------------------------
def create_hybrid_search_requests(dense_vector,
                                  sparse_vector,
                                  dense_params=None,
                                  sparse_params=None,
                                  expr=None,
                                  expr_params=None,
                                  limit=5) -> List[AnnSearchRequest]:
    """
    创建混合搜索请求

    :param dense_vector: 稠密向量
    :param sparse_vector: 稀疏向量
    :param dense_params: 稠密向量搜索参数，默认为None
    :param sparse_params: 稀疏向量搜索参数，默认为None
    :param expr: 查询表达式，默认为None
    :param expr_params: 查询表达式的变量内容，默认为None
    :param limit: 返回结果数量限制，默认为5
    :return: 包含稠密和稀疏搜索请求的列表
    :raises ValueError: 向量参数无效
    :raises RuntimeError: 创建请求失败
    """
    if dense_vector is None or sparse_vector is None:
        raise ValueError("dense_vector 和 sparse_vector 不能为 None")

    try:
        # 默认参数
        if dense_params is None:
            dense_params = {"metric_type": "COSINE"}
        if sparse_params is None:
            sparse_params = {"metric_type": "IP"}

        # 创建稠密向量搜索请求
        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param=dense_params,
            expr=expr,  # 过滤的条件(表达式写法)
            expr_params=expr_params,
            limit=limit
        )

        # 创建稀疏向量搜索请求
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param=sparse_params,
            expr=expr,
            expr_params=expr_params,
            limit=limit
        )

        return [dense_req, sparse_req]
    except Exception as e:
        raise RuntimeError(f"创建混合搜索请求失败: {e}") from e


# ------------------------------------------------------------------
# 执行混合检索请求
# ------------------------------------------------------------------
def execute_hybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_requests,
                                ranker_weights=(0.5, 0.5),
                                norm_score=True,
                                limit=5,
                                output_fields=None,
                                search_params=None):
    """
    执行混合搜索

    :param milvus_client: Milvus客户端
    :param collection_name: 集合名称
    :param search_requests: 搜索请求列表，通常是[dense_req, sparse_req]
    :param ranker_weights: 权重排名器的权重，默认为(0.5, 0.5)
    :param norm_score: 是否对分数进行归一化，默认为True
    :param limit: 返回结果数量限制，默认为5
    :param output_fields: 要返回的字段列表，默认为None
    :param search_params: 搜索参数，默认为None
    :return: 搜索结果
    :raises ValueError: 参数无效
    :raises RuntimeError: 搜索执行失败
    """
    if milvus_client is None:
        raise ValueError("milvus_client 不能为 None")
    if search_requests is None or len(search_requests) == 0:
        raise ValueError("search_requests 不能为 None 或空列表")

    try:
        # 创建权重融合排序器
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 默认输出字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行搜索
        res = milvus_client.hybrid_search(
            collection_name=collection_name,
            reqs=search_requests,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        # 动态计算所有查询返回的结果总数
        total_hits = sum(len(hits) for hits in res) if res else 0
        logger.info(f"Milvus 混合搜索完成，共处理 {len(res) if res else 0} 个查询，总计找到 {total_hits} 个结果")
        return res
    except Exception as e:
        raise RuntimeError(f"执行Milvus混合搜索失败 (collection={collection_name}): {e}") from e


def _item_names_filter(item_names: List[str]) -> Tuple[str, Dict[str, Any]]:
    expr = "item_name in {item_names}"
    expr_params = {"item_names": item_names}
    return expr, expr_params
