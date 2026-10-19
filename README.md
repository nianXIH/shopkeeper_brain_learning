# 小晞智库（shopkeeper_brain）

基于 **LangGraph + RAG** 的商品知识库问答系统。项目由两个独立的 FastAPI 服务组成：

- **导入服务（端口 8000）**：上传商品说明书（PDF / Markdown），经 MinerU 解析、切片、商品名识别、向量化后写入 Milvus，原始文件与图片存入 MinIO。
- **查询服务（端口 8001）**：接收用户提问，通过商品名确认、多路混合检索（向量 / HyDE / MCP Web 搜索）、RRF 融合、重排序、LLM 生成答案，支持 SSE 流式输出与多轮会话历史。

两个服务均自带静态前端页面（上传页 / 对话页 / 导航首页），开箱即用。

## 功能特性

- **文档导入流水线**：PDF → Markdown（MinerU）→ 图片提取与 VLM 总结 → 文档切片 → 商品名识别（LLM）→ 向量嵌入（BGE-M3）→ 写入 Milvus
- **查询流水线**：
  - 商品名智能确认：高置信度直接放行，模糊时返回候选选项让用户选择
  - 多路并行检索：稠密向量检索 + HyDE 假设文档检索 + DashScope MCP Web 搜索
  - RRF（Reciprocal Rank Fusion）融合多路结果
  - BGE 重排序，动态截断 Top-K
  - LLM 生成最终答案
- **流式输出**：SSE 协议逐字推送答案；非流式接口同步返回
- **会话历史**：MongoDB 存储多轮对话记录，支持查询与清空
- **异步任务**：导入流程后台执行，前端轮询任务状态（解析 PDF 耗时较长）
- **对象存储**：上传文件与解析产物持久化到 MinIO

## 技术栈

| 类别 | 组件 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn |
| 工作流编排 | LangGraph |
| PDF 解析 | MinerU（mineru[all]，本地引擎） |
| LLM / VLM | OpenAI 兼容 API（阿里云 DashScope，qwen 系列） |
| 嵌入模型 | BAAI/bge-m3（本地，sentence-transformers，支持 GPU / FP16） |
| 重排序模型 | BAAI bge-reranker-large（本地，支持 GPU / FP16） |
| 向量数据库 | Milvus |
| 文档数据库 | MongoDB（会话历史） |
| 对象存储 | MinIO |
| Web 搜索 | DashScope WebSearch MCP（openai-agents + mcp 1.x） |
| 流式协议 | SSE（Server-Sent Events） |

## 系统架构

### 导入流程（import_processor）

```text
entry_node（判断文件类型）
    │
    ├── PDF ──> pdf_to_md_node ──┐
    │                            │
    └── MD ──────────────────────┤
                                 v
                        md_to_img_node（图片提取 + VLM 总结）
                                 v
                       document_split_node（文档切片）
                                 v
                  item_name_recognition_node（商品名识别）
                                 v
                     embedding_chunks_node（向量嵌入）
                                 v
                      import_milvus_node（写入 Milvus）
```

### 查询流程（query_processor）

```text
item_name_confirmed_node（商品名确认）
    │
    ├──（已有答案/需用户选择）──> answer_output_node ──> END
    │
    └──（无答案）
        v
    multi_search（并行分发）
        │
   ┌────┼──────────────┐
   v    v              v
embedding  hyde    web_mcp
   │    │              │
   └────┼──────────────┘
        v
      join（汇合）
        v
  rrf_merge_node（RRF 融合）
        v
  reranker_node（BGE 重排序）
        v
  answer_output_node（LLM 生成答案）
        v
       END
```

## 目录结构

```text
knowledge/
├── api/                    # FastAPI 路由层
│   ├── import_router.py    # 导入服务（端口 8000）：上传、任务状态
│   └── query_router.py     # 查询服务（端口 8001）：查询、SSE 流、会话历史
├── core/                   # 路径解析、依赖注入
├── front/                  # 静态前端页面（index / import / chat）
├── processor/
│   ├── import_processor/   # 导入流水线（LangGraph）
│   │   └── nodes/          # 各处理节点
│   └── query_processor/    # 查询流水线（LangGraph）
│       └── nodes/          # 各处理节点
├── prompts/                # LLM 提示词（导入 / 查询）
├── schema/                 # 请求 / 响应数据模型
├── service/                # 业务服务层（上传、查询）
├── utils/                  # 工具类（Milvus / MongoDB / MinIO / SSE / 任务等）
├── test/                   # 各组件独立测试脚本
├── temp_data/              # 上传与解析的临时产物（按日期/任务分目录）
├── .env                    # 环境变量配置（不入库）
└── requirements.txt        # Python 依赖
```

## 快速开始

### 环境要求

- Python 3.12+
- 可用的 Milvus、MongoDB、MinIO 实例
- DashScope（或任意 OpenAI 兼容）API Key
- 本地 BGE-M3 / bge-reranker 模型文件（GPU 可选但推荐）

### 安装依赖

虚拟环境位于 `knowledge/.venv`（注意：不在项目根目录下）：

```bash
cd knowledge
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> 注意：`torch==2.14.1+cu130` / `torchvision==0.29.1+cu130` 为 CUDA 版本，需从对应 index 安装；若安装了 `mineru[all]` 后 GPU 不可用，需重装 GPU 版 PyTorch。

### MinerU 准备

导入服务依赖 MinerU 本地解析引擎：

```bash
# 下载标准档位模型（ModelScope 源）
mineru-kit models download --tier standard --source modelscope
# 验证模型完整性
mineru-kit models verify --tier standard
# 查看模型与配置状态（配置文件位于 ~/.mineru/config.yaml）
mineru-kit models show
# 启动本地解析服务（新版 CLI 必须先启动服务才能 parse）
mineru server start
```

### 配置环境变量

在 `knowledge/` 下创建 `.env` 文件，主要配置项：

```ini
# LLM（OpenAI 兼容 / DashScope）
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen3.8-flash
VL_MODEL=qwen3.8-flash          # 图片总结 VLM
ITEM_MODEL=qwen-flash           # 商品名识别

# BGE 本地模型
BGE_M3_PATH=D:\ai_models\modelscope_cache\models\BAAI\bge-m3
BGE_DEVICE=cuda:0
BGE_FP16=True
BGE_RERANKER_LARGE=D:\ai_models\modelscope_cache\models\BAAI
BGE_RERANKER_DEVICE=cuda:0

# Milvus
MILVUS_URL=http://<host>:19530
CHUNKS_COLLECTION=kb_chunks_v1
ITEM_NAME_COLLECTION=kb_item_names_v1
MILVUS_METRIC_TYPE=COSINE
MILVUS_MIN_COSINE_SCORE=0.75

# MongoDB（会话历史）
MONGO_URL=mongodb://user:pass@<host>:27017
MONGO_DB_NAME=kb001

# MinIO（对象存储）
MINIO_ENDPOINT=<host>:9000
MINIO_ACCESS_KEY=xxx
MINIO_SECRET_KEY=xxx
MINIO_BUCKET_NAME=knowledge-base-v1

# MCP Web 搜索
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp

# 模型缓存 / 临时目录
MODELSCOPE_CACHE=D:/ai_models/modelscope_cache
HF_HOME=D:/ai_models/huggingface_cache
MD_ROOT_DIR=./temp-files/
```

检索、重排序、RRF、商品名确认等流程参数（如 `RERANK_MAX_TOP_K`、`RRF_K`、`ITEM_NAME_HIGH_CONFIDENCE`）均有合理默认值，可通过环境变量覆盖，详见 [processor/query_processor/config.py](processor/query_processor/config.py) 与 [processor/import_processor/config.py](processor/import_processor/config.py)。

### 启动服务

```bash
# 导入服务（端口 8000）
python -m knowledge.api.import_router

# 查询服务（端口 8001）
python -m knowledge.api.query_router
```

启动后访问：

- 导航首页：http://localhost:8000/front/ （或 8001，均可跳转）
- 导入页：http://localhost:8000/front/import.html
- 对话页：http://localhost:8001/front/chat.html
- 接口文档：http://localhost:8000/docs 、http://localhost:8001/docs

## API 概览

### 导入服务（:8000）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/upload` | 上传 PDF / MD 文件，返回 `task_id`，后台执行导入流水线 |
| GET | `/status/{task_id}` | 查询导入任务状态（前端约 1.5s 轮询一次） |

### 查询服务（:8001）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/query` | 提交查询；`is_stream=true` 返回 `task_id` 供 SSE 订阅，否则同步返回答案 |
| GET | `/stream/{task_id}` | SSE 流式接收答案 |
| GET | `/history/{session_id}?limit=50` | 获取会话历史（1-200 条） |
| DELETE | `/history/{session_id}` | 清空指定会话历史 |

## 测试

`test/` 目录下包含各组件的独立验证脚本（MinerU 解析、BGE 嵌入 / 重排序、Milvus、MinIO、MCP、表格处理等），可在接入主流程前单独运行验证环境连通性。
