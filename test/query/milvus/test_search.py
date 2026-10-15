"""
Milvus 三种向量检索演示
支持：稠密向量检索、稀疏向量检索、混合向量检索
"""

from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import time

# ─────────────────────────────────────────────────────────
# 1. 演示数据集
# ─────────────────────────────────────────────────────────

DEMO_DOCUMENTS = [
    {"id": 1, "title": "Python异步编程指南", "content": "介绍async/await语法、asyncio库的使用...", "category": "Python"},
    {"id": 2, "title": "JavaScript异步编程详解", "content": "Promise和async函数、事件循环...", "category": "JavaScript"},
    {"id": 3, "title": "深入理解Python装饰器", "content": "装饰器原理与实战应用...", "category": "Python"},
    {"id": 4, "title": "React Hooks入门教程", "content": "useState和useEffect详解...", "category": "React"},
    {"id": 5, "title": "Vue3组合式API详解", "content": "setup函数和响应式系统...", "category": "Vue"},
]

COLLECTION_NAME = "demo_tech_articles"


# ─────────────────────────────────────────────────────────
# 2. 创建 Collection
# ─────────────────────────────────────────────────────────


def create_collection(client: MilvusClient):
    """创建支持混合检索的 Collection"""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(enable_dynamic_field=True)

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    print(f"Collection '{COLLECTION_NAME}' 创建成功")


# ─────────────────────────────────────────────────────────
# 3. 生成向量并插入数据
# ─────────────────────────────────────────────────────────

def generate_and_insert_data(client, embedding_model):
    """生成混合向量并插入数据"""
    texts = [f"{doc['title']}\n{doc['content']}" for doc in DEMO_DOCUMENTS]
    embedding_result = embedding_model.encode_documents(texts)

    # 提取稠密向量
    dense_vectors = [vec.tolist() for vec in embedding_result['dense']]

    # 提取稀疏向量
    sparse_vectors = []
    csr_array = embedding_result['sparse']
    for i in range(len(texts)):
        start, end = csr_array.indptr[i], csr_array.indptr[i + 1]
        token_ids = csr_array.indices[start:end].tolist()
        weights = csr_array.data[start:end].tolist()
        sparse_vectors.append(dict(zip(token_ids, weights)))

    # 构建插入数据
    data = []
    for i, doc in enumerate(DEMO_DOCUMENTS):
        data.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "category": doc["category"],
            "dense_vector": dense_vectors[i],
            "sparse_vector": sparse_vectors[i]
        })

    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"插入 {result['insert_count']} 条数据")
    return dense_vectors, sparse_vectors


# ─────────────────────────────────────────────────────────
# 4. 三种检索方式
# ─────────────────────────────────────────────────────────

def dense_search(client, query_dense, limit=3):
    """稠密向量检索：语义理解强"""
    dense_result = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_dense],
        anns_field="dense_vector",
        search_params={"metric_type": "COSINE"},
        limit=limit,
        output_fields=["title", "category"]
    )
    print(f"稠密向量检索的结果:{dense_result}")
    return dense_result[0]


def sparse_search(client, query_sparse, limit=3):
    """稀疏向量检索：关键词精确匹配"""
    sparse_result = client.search(
        collection_name=COLLECTION_NAME, data=[query_sparse],
        anns_field="sparse_vector", search_params={"metric_type": "IP"},
        limit=limit,
        output_fields=["title", "category"]
    )
    print(f"稀疏向量检索的结果:{sparse_result}")
    return sparse_result[0]


def hybrid_search(client, query_dense, query_sparse, limit=3):
    """混合向量检索：语义+关键词"""
    dense_req = AnnSearchRequest(data=[query_dense],
                                 anns_field="dense_vector",
                                 param={"metric_type": "COSINE"},
                                 limit=limit)
    sparse_req = AnnSearchRequest(data=[query_sparse],
                                  anns_field="sparse_vector",
                                  param={"metric_type": "IP"},
                                  limit=limit)
    reranker = WeightedRanker(0.5, 0.5, norm_score=True)
    hybrid_result = client.hybrid_search(
        collection_name=COLLECTION_NAME, reqs=[dense_req, sparse_req],
        ranker=reranker, limit=limit, output_fields=["title", "category"]
    )
    print(f"混合向量检索的结果:{hybrid_result}")
    return hybrid_result[0]


# ─────────────────────────────────────────────────────────
# 5. 主程序
# ─────────────────────────────────────────────────────────

def main():
    from knowledge.utils.client.ai_clients import AIClients
    print("=" * 50)
    print("Milvus 三种向量检索演示")
    print("=" * 50)

    client = MilvusClient(uri="http://192.168.253.140:19530")
    embedding_model = AIClients.get_bge_m3_client()

    create_collection(client)
    generate_and_insert_data(client, embedding_model)

    time.sleep(2)  # 等待索引生效

    # 测试查询
    query_text = "Python异步编程如何使用"
    print(f"\n查询: '{query_text}'")

    query_emb = embedding_model.encode_documents([query_text])
    query_dense = query_emb['dense'][0].tolist()
    csr = query_emb['sparse']
    query_sparse = dict(zip(csr.indices[csr.indptr[0]:csr.indptr[1]].tolist(),
                            csr.data[csr.indptr[0]:csr.indptr[1]].tolist()))

    # 执行检索
    # print("\n【稠密向量检索】")
    # for hit in dense_search(client, query_dense):
    #     print(f"  {hit['entity']['title']} ---> {hit['distance']:.4f}")

    # print("\n【稀疏向量检索】")
    # for hit in sparse_search(client, query_sparse):
    #     print(f"  {hit['entity']['title']} - {hit['distance']:.4f}")

    print("\n【混合向量检索】")
    for hit in hybrid_search(client, query_dense, query_sparse):
        print(f"  {hit['entity']['title']} - {hit['distance']:.4f}")


if __name__ == "__main__":
    main()
