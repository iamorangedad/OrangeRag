# 结构化日志系统使用指南

## 概述

OrangeRag 现已支持结构化日志（JSON格式），便于日志聚合、分析和可视化。系统支持通过环境变量配置日志级别、格式和输出目标。

## 快速开始

### 默认行为
无需任何配置，系统将自动使用 JSON 格式输出日志到控制台，日志级别为 INFO。

```bash
cd backend
python main.py
```

### 配置日志级别

```bash
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
python main.py
```

### 切换为文本格式

```bash
export LOG_FORMAT=text  # json 或 text
python main.py
```

### 输出到文件

```bash
export LOG_OUTPUT=file
export LOG_FILE_PATH=/var/log/orangerag/app.log
python main.py
```

## 环境变量配置

### 基础配置

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `LOG_LEVEL` | `INFO` | 日志级别: DEBUG, INFO, WARNING, ERROR |
| `LOG_FORMAT` | `json` | 日志格式: json, text |
| `LOG_OUTPUT` | `console` | 输出目标: console, file, both |

### 文件日志配置（当 `LOG_OUTPUT=file` 或 `both` 时有效）

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `LOG_FILE_PATH` | `/var/log/orangerag/app.log` | 日志文件路径 |
| `LOG_FILE_MAX_BYTES` | `104857600` | 单个日志文件最大大小（字节），默认100MB |
| `LOG_FILE_BACKUP_COUNT` | `10` | 保留的备份文件数量 |

### 模块级别配置

可以为特定模块设置不同的日志级别：

```bash
export LOG_LEVEL_MODULE_OLLAMA=WARNING
export LOG_LEVEL_MODULE_RETRIEVERS=DEBUG
export LOG_LEVEL_MODULE_HYBRID_CHAT_SERVICE=INFO
```

环境变量格式：`LOG_LEVEL_MODULE_{MODULE_NAME}`

模块名称转换规则：
- 环境变量: `LOG_LEVEL_MODULE_OLLAMA`
- 模块名: `ollama`
- 完整路径: `app.core.llm.ollama`

## 日志格式示例

### JSON 格式

```json
{
  "timestamp": "2026-02-15T10:30:45.123Z",
  "level": "INFO",
  "module": "app.services.hybrid_chat_service",
  "message": "HybridChat citation_retrieval completed in 1250.50ms",
  "service": "HybridChat",
  "stage": "citation_retrieval",
  "duration_ms": 1250.5,
  "conversation_id": "abc-123"
}
```

### 文本格式

```
2026-02-15 10:30:45,123 [INFO] [app.services.hybrid_chat_service] [HybridChat] [citation_retrieval] [conv:abc-123] (1250.50ms) HybridChat citation_retrieval completed in 1250.50ms
```

### 错误日志

```json
{
  "timestamp": "2026-02-15T10:30:45.456Z",
  "level": "ERROR",
  "module": "app.services.chat_service",
  "message": "Chat error: LLM request timed out",
  "service": "Chat",
  "error_type": "TimeoutError",
  "error_message": "LLM request timed out after 180s",
  "stack_trace": "Traceback (most recent call last):...",
  "conversation_id": "abc-123"
}
```

## 在代码中使用

### 基本用法

```python
from app.core.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Application started")
```

### 记录性能指标

```python
from app.core.logging_config import log_performance

# 记录性能指标
log_performance(
    logger,
    service="HybridChat",           # 服务名称
    stage="retrieval",              # 处理阶段
    duration_ms=1250.5,             # 耗时（毫秒）
    conversation_id="conv-123",     # 可选：会话ID
    task_id="task-456",             # 可选：任务ID
    extra={"results_found": 10}     # 可选：额外字段
)
```

### 记录错误

```python
from app.core.logging_config import log_error

try:
    # 业务逻辑
    result = process_data()
except Exception as e:
    log_error(
        logger,
        service="IndexingService",      # 服务名称
        error=e,                        # 异常对象
        task_id="task-789",             # 可选：任务ID
        conversation_id="conv-123"      # 可选：会话ID
    )
    raise
```

### 添加自定义字段

```python
# 在日志中添加自定义字段
logger.info(
    "Processing completed",
    extra={
        "custom_field": "value",
        "user_id": "user-123"
    }
)
```

## 生产环境推荐配置

### Docker 环境

```dockerfile
# Dockerfile
ENV LOG_LEVEL=INFO
ENV LOG_FORMAT=json
ENV LOG_OUTPUT=file
ENV LOG_FILE_PATH=/var/log/orangerag/app.log
ENV LOG_FILE_MAX_BYTES=104857600
ENV LOG_FILE_BACKUP_COUNT=10
```

### Kubernetes 环境

```yaml
# deployment.yaml
env:
  - name: LOG_LEVEL
    value: "INFO"
  - name: LOG_FORMAT
    value: "json"
  - name: LOG_OUTPUT
    value: "both"  # 同时输出到控制台和文件
  - name: LOG_FILE_PATH
    value: "/var/log/orangerag/app.log"
  - name: LOG_LEVEL_MODULE_OLLAMA
    value: "WARNING"
volumeMounts:
  - name: logs
    mountPath: /var/log/orangerag
volumes:
  - name: logs
    emptyDir: {}
```

### 日志收集（Fluentd/Fluent Bit）

```yaml
# fluent-bit.conf
[INPUT]
    Name tail
    Path /var/log/orangerag/app.log
    Parser json
    Tag orangerag.app

[OUTPUT]
    Name elasticsearch
    Match orangerag.*
    Host elasticsearch
    Port 9200
    Index orangerag-logs
```

## 开发环境推荐配置

```bash
# .env 文件
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LOG_OUTPUT=console
```

## 故障排查

### 日志不输出

1. 检查日志级别设置
   ```bash
   echo $LOG_LEVEL
   ```

2. 确认日志目录权限（文件输出模式）
   ```bash
   mkdir -p /var/log/orangerag
   chmod 755 /var/log/orangerag
   ```

### JSON 格式解析失败

确保使用 JSON 格式时，日志消息本身不包含未转义的特殊字符。

### 模块日志级别不生效

检查模块名称拼写：
```python
# 正确
export LOG_LEVEL_MODULE_OLLAMA=WARNING

# 错误（应为 ollama，不是 llm）
export LOG_LEVEL_MODULE_LLM_OLLAMA=WARNING
```

## 最佳实践

1. **始终使用 get_logger**: 不要直接使用 `logging.getLogger(__name__)`
   ```python
   from app.core.logging_config import get_logger
   logger = get_logger(__name__)
   ```

2. **使用辅助函数记录性能**: 使用 `log_performance` 而不是手动格式化
   ```python
   # 推荐
   log_performance(logger, "Service", "stage", duration_ms)
   
   # 不推荐
   logger.info(f"Completed in {duration_ms}ms")
   ```

3. **为关键操作添加上下文**: 包括 conversation_id 或 task_id
   ```python
   log_performance(
       logger, "HybridChat", "retrieval", duration_ms,
       conversation_id=conv_id  # 添加上下文
   )
   ```

4. **在生产环境使用 JSON 格式**: 便于日志聚合和分析

5. **合理设置日志级别**:
   - 开发环境: DEBUG
   - 测试环境: INFO
   - 生产环境: INFO 或 WARNING

## 与 ELK Stack 集成

### Filebeat 配置

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/orangerag/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "orangerag-logs-%{+yyyy.MM.dd}"
```

### Kibana 查询示例

```
# 查询特定服务的慢请求
service:HybridChat AND duration_ms > 1000

# 查询特定会话的所有日志
conversation_id:abc-123

# 查询错误日志
level:ERROR

# 查询特定阶段的性能
stage:retrieval AND duration_ms > 500
```

---

*文档版本: 1.0*  
*最后更新: 2026-02-15*
