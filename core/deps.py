from functools import cache, lru_cache

from knowledge.service.upload_service import UpLoadService


@cache  # 缓存注解（将实例对象缓存一份:可能会出现oom:out of memory）实现单例效果
# @lru_cache # 缓存注解(淘汰机制：当缓存的数据量达到一定的阈值 根据lru算法将其缓存数据淘汰掉)
def get_upload_file_service():
    return UpLoadService()
