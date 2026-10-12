from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# bge3-m3嵌入模型默认输出的稠密向量维度是1024
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='D:\\ai_models\\modelscope_cache\\models\\BAAI\\bge-m3',
    device='cuda:0',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=True  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.  # 半精度  单精度
)

# 嵌入对象（query or document）encode_queries  # 用户问题嵌入（批量文档） or encode_documents() # 文档内容嵌入（一个文档）

vector_result = bge_m3_ef.encode_queries(queries=['我是中国人','你是美国人'])

# 1. 稠密向量
print(vector_result.get('dense')[0].tolist())

# 2. 稀疏向量(token_id: 权重)--->{"token_id":"权重","token_id":"权重"，"token_id":权重}---》Milvus用户稀疏向量的结构必须是一个字典且key:必须是token_id value：必须是权重
sparse_array=vector_result.get('sparse')
print(sparse_array)

