from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    model_name_or_path=r"D:\\ai_models\\modelscope_cache\\models\\BAAI",
    device="cuda",
    use_fp16=True
)

# 计算相关性得分
pairs = [
    ("什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"),
    ("什么是万用表？", "今天天气很好")
]

scores = reranker.compute_score(sentence_pairs=("什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"))  # [分数1,分数2]:只会计算q-d得分，排序交给业务层
print(scores)