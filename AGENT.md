# OrangeRag 项目智能助手文档

本文档用于快速理解项目架构，避免重复阅读整个代码库。

## 一、项目概述

**项目名称**: OrangeRag  
**类型**: 基于 FastAPI + LlamaIndex + Ollama 的智能文档问答系统（RAG）  
**部署方式**: 本地开发 / Docker / Kubernetes

### 核心特性
- 混合RAG检索：Dense(向量) + Sparse(BM25) + RRF融合 + 可选重排序
- 多向量存储支持：Simple(内存) / ChromaDB
- 模块化架构：检索器、融合算法、向量存储、LLM均有抽象基类
- A/B测试支持
- 异步任务处理

---

## 二、项目架构

```
/home/jetson/OrangeRag/
├── backend/                    # 后端服务（FastAPI）
│   ├── main.py                 # 主入口文件
│   ├── app/
│   │   ├── api/               # API层 - HTTP路由
│   │   │   └── routes/
│   │   │       ├── chat.py           # 聊天API
│   │   │       ├── documents.py      # 文档管理API
│   │   │       ├── health.py         # 健康检查API
│   │   │       ├── models.py         # 模型管理API
│   │   │       └── tasks.py          # 异步任务API
│   │   ├── core/              # 核心抽象层
│   │   │   ├── cache/         # 缓存模块（BM25）
│   │   │   ├── citation/      # 引用检索模块
│   │   │   ├── document_processing/  # 文档处理（PDF提取）
│   │   │   ├── fusion/        # 融合算法（RRF）
│   │   │   ├── llm/           # LLM抽象层
│   │   │   ├── metadata/      # 元数据处理
│   │   │   ├── prompt/        # 提示词构建
│   │   │   ├── query/         # 查询扩展
│   │   │   ├── reranker/      # 重排序模块
│   │   │   ├── retrievers/    # 检索器（混合RAG）
│   │   │   │   ├── base.py
│   │   │   │   ├── dense_retriever.py
│   │   │   │   ├── sparse_retriever.py
│   │   │   │   └── hybrid_retriever.py
│   │   │   ├── testing/       # A/B测试
│   │   │   └── vector_store/  # 向量存储抽象
│   │   │       ├── base.py
│   │   │       ├── chroma.py
│   │   │       └── simple.py
│   │   ├── models/            # 数据模型（Pydantic）
│   │   ├── services/          # 业务逻辑层
│   │   │   ├── chat_service.py
│   │   │   ├── hybrid_chat_service.py  # 主聊天服务
│   │   │   ├── document_service.py
│   │   │   ├── indexing_service.py     # 异步索引
│   │   │   ├── large_document_processor.py
│   │   │   └── model_service.py
│   │   └── utils/             # 工具函数
│   └── tests/
├── frontend/                   # 前端静态文件（index.html）
├── deployment/                 # Kubernetes部署配置
├── docs/                       # 项目文档
└── Dockerfile
```

---

## 三、关键配置

### 启动命令
```bash
# 开发模式
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 核心配置文件
| 文件 | 用途 |
|------|------|
| `backend/pyproject.toml` | Python项目配置、依赖、工具配置 |
| `backend/app/config.py` | 运行时配置（Pydantic Settings） |
| `backend/pytest.ini` | 测试配置 |

### 环境变量
```python
debug: bool = Field(default=False, alias="DEBUG")
ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
default_model_name: str = Field(default="qwen2.5:14b", alias="DEFAULT_MODEL_NAME")
```

---

## 四、日志系统现状

### 基础配置
- **位置**: `backend/main.py` 第22-27行
- **模块**: Python标准库 `logging`
- **级别**: INFO
- **格式**: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`

### 日志分布（197处调用）
| 模块 | 级别 | 用途 |
|------|------|------|
| `main.py` | INFO | 启动信息、服务状态 |
| `hybrid_chat_service.py` | INFO/DEBUG/WARNING/ERROR | 混合RAG流程（含性能日志） |
| `indexing_service.py` | INFO/WARNING/ERROR | 异步索引任务 |
| `chat_service.py` | INFO/ERROR | 聊天服务 |
| `large_document_processor.py` | INFO/WARNING/ERROR | 大文档处理 |
| `citation/retriever.py` | INFO/WARNING/DEBUG | 引用检索 |
| `metadata/matcher.py` | INFO/DEBUG | 元数据匹配 |
| `document_processing/pdf_extractor.py` | INFO/DEBUG/ERROR | PDF处理 |

### 模块前缀标识
所有服务使用 `[ServiceName]` 前缀便于过滤：
- `[HybridChat]` - 混合RAG聊天服务
- `[AsyncIndexing]` - 异步索引服务
- `[Chat]` - 基础聊天服务
- `[Upload]`, `[Delete]` - 文档服务
- `[RobustProcessor]` - 大文档处理器
- `[CitationRetriever]` - 引用检索器
- `[MetadataMatcher]` - 元数据匹配器

### 性能日志示例
`hybrid_chat_service.py` 实现了5阶段性能监控：
```python
# 阶段1：元数据匹配
# 阶段2：引用检索（记录耗时）
# 阶段3：混合RAG检索（记录耗时）
# 阶段4：提示构建
# 阶段5：LLM生成（记录耗时）
# 总计时间
```

### 当前优缺点
**优点**:
- 日志覆盖全面
- 有性能耗时监控
- 错误堆栈跟踪完整
- 统一模块前缀标识

**缺点**:
- 无结构化日志（JSON格式）
- 无请求追踪ID（correlation ID）
- 日志配置硬编码
- 无全局异常处理
- 无自定义异常类

---

## 五、错误处理现状

### API路由层
每个路由函数独立处理异常，使用 HTTPException：
- `chat.py` - 500 (通用), ValueError特殊处理
- `documents.py` - 400 (ValueError), 404 (FileNotFound), 500 (通用)
- `tasks.py` - 404 (任务不存在), 500 (通用)

### 服务层
- 关键错误捕获并记录堆栈跟踪
- 使用 `import traceback` 运行时导入（非最佳实践）

### 缺失
- 全局异常处理器
- 自定义异常类层次结构
- 统一的错误响应格式

---

## 六、中间件配置

### 已配置
- **CORS中间件**: `main.py` 第135-141行

### 未配置
- 请求日志中间件
- 请求追踪ID中间件
- 慢请求告警中间件

---

## 七、关键文件速查

| 功能 | 文件路径 |
|------|---------|
| 主入口 | `backend/main.py` |
| 应用配置 | `backend/app/config.py` |
| 混合RAG服务 | `backend/app/services/hybrid_chat_service.py` |
| 异步索引 | `backend/app/services/indexing_service.py` |
| 文档服务 | `backend/app/services/document_service.py` |
| 混合检索器 | `backend/app/core/retrievers/hybrid_retriever.py` |
| LLM实现 | `backend/app/core/llm/ollama.py` |
| 聊天API | `backend/app/api/routes/chat.py` |
| 文档API | `backend/app/api/routes/documents.py` |

---

## 八、优化计划

### 方案1：结构化日志系统（已完成）
- 创建日志配置模块 `backend/app/core/logging_config.py`
- 支持 JSON 和 Text 两种格式
- 支持环境变量配置（LOG_LEVEL, LOG_FORMAT, LOG_OUTPUT 等）
- 支持日志轮转（RotatingFileHandler）
- 支持模块级别日志级别控制
- 提供 `log_performance` 和 `log_error` 辅助函数
- 更新 main.py 使用新的日志配置
- 修改服务层使用结构化日志
- 添加单元测试 `backend/tests/unit/test_logging_config.py`
- 编写使用文档 `docs/LOGGING_GUIDE.md`

**快速使用：**
```bash
# JSON 格式输出（默认）
python backend/main.py

# 文本格式
export LOG_FORMAT=text
python backend/main.py

# 输出到文件
export LOG_OUTPUT=file
export LOG_FILE_PATH=/var/log/orangerag/app.log
python backend/main.py
```

### 方案2：请求追踪系统（待规划）

### 方案3：全局异常处理（待规划）

### 方案4：增强型中间件（待规划）

### 方案5：配置化管理（待规划）

---

*最后更新: 2026-02-15*
