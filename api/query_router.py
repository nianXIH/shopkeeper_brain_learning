import asyncio
import logging

import uvicorn, os
from typing import Union
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks, Request,HTTPException, Query as FastQuery
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.core.deps import get_query_service
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_util import get_task_status, TASK_STATUS_FAILED


def create_app():
    """
    创建FastAPI实例
    Returns:
    FastAPI实例

    """

    # 1. 实例化
    app = FastAPI(description="小晞智库查询的应用", version="v1.0")

    # 1.1 应用入口统一配置根日志（各业务模块只getLogger，不再各自basicConfig）
    logging.basicConfig(level=logging.INFO)

    # 2. 跨域配置：来源白名单走环境变量CORS_ALLOW_ORIGINS（逗号分隔），
    #    默认*仅适用于本地开发，生产环境应显式收紧
    allow_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,  # ← 和 credentials=True 冲突
        allow_credentials=False,  # 自定义cookies Authorization  tsl客户端证书信息
        allow_methods=["*"],  # ← 和 credentials=True 冲突  GET(获取资源)  POST(新增)  DELETE（删除） PUT（修改）
        allow_headers=["*"],  # ← 和 credentials=True 冲突   自定义的头字段 token  content-type:application/json
    )

    # 3. 挂载静态文件（import.html）
    page_dir = get_front_page_dir()

    if page_dir and os.path.exists(page_dir):
        # / 与 /front/ 统一导向导航首页 index.html，由它跳转到上传页(8000)或查询页(8001)
        @app.get("/", include_in_schema=False)
        async def root_index():
            return RedirectResponse(url="/front/")

        @app.get("/front/", include_in_schema=False)
        async def front_index():
            return RedirectResponse(url="/front/index.html")

        app.mount("/front", StaticFiles(directory=page_dir), name="front")

    # 4. 注册路由
    register_router(app)
    # 5. 返回fastapi实例
    return app
def register_router(app: FastAPI):
    @app.get("/api", include_in_schema=False)
    def hello_world():
        # 根路径 / 已重定向到前端导航页，这里保留一个 /api 健康检查入口
        return {"flag": "success"}

    @app.post("/query")
    async def query(request: QueryRequest,
                    background_tasks: BackgroundTasks,
                    service: QueryService = Depends(get_query_service),
                    ) -> Union[
        StreamSubmitResponse, QueryResponse]:  # FastAPI:自动将json格式的字符串反序列化为指定的约束schema
        """
        处理查询请求
        Args:
            request: 前端发送的请求数据（服务端用对象接收）
            background_tasks： FastAPI提供的后台任务对象
            service： 查询业务组件对象


        Returns:

        """

        # 1. 获取session_id
        session_id = request.session_id or service.generate_session_id()

        # 2. 获取任务id
        task_id = service.generate_task_id()

        # 3. 如果是流式调用
        # 3.1 流式调用
        if request.is_stream:
            # a. sse队列创建出来(容器中任务id和队列的对应关系也有了)
            create_sse_queue(task_id=task_id)

            # b. 利用fastapi的background启动的线程执行查询任务
            background_tasks.add_task(service.run_query_graph,
                                      session_id=session_id,
                                      task_id=task_id,
                                      query=request.query,
                                      is_stream=request.is_stream)
            return StreamSubmitResponse(message="查询请求已经提交", session_id=session_id, task_id=task_id)

        # 3.2 非流式调用(直接用当前线程运行查询流程 不启动一个新线程执行--->合理?。不合理：不是只有流式慢 非流式也慢 )
        else:

            # a. 运行查询的pineline
            # 获取当前事件循环对象(uvicorn)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, service.run_query_graph, session_id, task_id, request.query,
                                       request.is_stream)
            # b. 任务失败时明确报错：run_query_graph 内部会吞掉异常只更新任务状态，
            #    不在这里拦截就会返回"200 + 空答案"，调用方无法区分"没答案"和"流程挂了"
            if get_task_status(task_id) == TASK_STATUS_FAILED:
                raise HTTPException(status_code=500, detail=f"查询任务执行失败 task_id={task_id}")
            # c. 从任务结果队列中获取答案
            answer = service.get_task_result(task_id)
            # d. 返回查询响应结果对象
            return QueryResponse(message="查询请求已经处理完了", session_id=session_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """
        返回sse协议要的数据包：流式+yield使用 最佳搭配
        1. 如何返回（直接返回）利用生成器yield返回(未来给的数据包通过yield返回)
        2. 返回注意事项："event:自定义\ndata:自定义\n\n"
        Args:
            task_id: 任务id
            StreamingResponse:将后端组合的sse协议格式的数据 返回给前端

        Returns:

        """
        return StreamingResponse(content=sse_generator(task_id, request), media_type="text/event-stream")




    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str,
            # limit加上限约束：不设上限时恶意请求可一次拉全量历史，拖垮MongoDB
            limit: int = FastQuery(50, ge=1, le=200, description="返回的历史条数(1-200)"),
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


if __name__ == '__main__':
    uvicorn.run(create_app(), host="0.0.0.0", port=8001)
