# 方案1：结构化日志系统开发计划

## 一、方案概述

**目标**: 引入 JSON 结构化日志系统，便于日志聚合、分析和可视化  
**优先级**: 高  
**预计工期**: 2-3天  
**依赖**: python-json-logger 或 structlog

### 关键收益
- 支持日志字段查询（如 `duration>1000 AND service=HybridChat`）
- 便于集成 ELK、Datadog 等日志分析平台
- 支持日志轮转和归档
- 可通过环境变量动态配置日志级别

---

## 二、技术选型

### 推荐方案：python-json-logger
**理由**:
- 轻量级，无需修改现有日志代码结构
- 与标准库 logging 完全兼容
- 支持自定义字段
- 社区活跃，维护良好

### 备选方案：structlog
**适用场景**:
- 需要更丰富的结构化功能
- 愿意重构现有日志代码

---

## 三、实施步骤

### 阶段1：基础配置（优先级：P0）
**任务1.1**: 添加日志配置文件
- 创建 `backend/app/core/logging_config.py`
- 实现 JSON 格式日志处理器
- 支持环境变量配置

**任务1.2**: 修改主入口
- 更新 `backend/main.py`
- 使用新的日志配置
- 保持向后兼容

**验收标准**:
- [ ] 日志输出为 JSON 格式
- [ ] 包含时间、级别、模块名、消息等基础字段
- [ ] 可通过 `LOG_LEVEL` 环境变量调整级别
- [ ] 可通过 `LOG_FORMAT` 环境变量切换格式（json/text）

---

### 阶段2：增强字段（优先级：P1）
**任务2.1**: 添加性能字段
- 在性能日志中添加结构化字段
- duration_ms, service, stage 等

**任务2.2**: 添加上下文字段
- conversation_id
- task_id
- request_id（预留）

**任务2.3**: 添加错误字段
- error_type
- error_message
- stack_trace

**验收标准**:
- [ ] 性能日志包含 duration_ms 字段
- [ ] 错误日志包含 error_type 和 stack_trace 字段
- [ ] 所有日志包含 conversation_id（如适用）

---

### 阶段3：日志轮转（优先级：P1）
**任务3.1**: 配置文件日志
- 添加 RotatingFileHandler
- 配置日志文件路径和大小限制

**任务3.2**: 生产环境配置
- 日志文件路径: `/var/log/orangerag/`
- 单个文件大小: 100MB
- 保留文件数: 10个

**任务3.3**: 开发环境配置
- 仅输出到控制台
- DEBUG 级别

**验收标准**:
- [ ] 生产环境日志写入文件
- [ ] 日志文件自动轮转
- [ ] 开发环境保持控制台输出

---

### 阶段4：模块级别控制（优先级：P2）
**任务4.1**: 模块日志级别配置
- 支持按模块设置不同日志级别
- 配置格式: `LOG_LEVEL_MODULE_{MODULE_NAME}`

**任务4.2**: 常用模块预配置
- `app.services.hybrid_chat_service` -> INFO
- `app.core.llm.ollama` -> WARNING
- `app.core.retrievers` -> DEBUG

**验收标准**:
- [ ] 可通过环境变量设置模块级别日志
- [ ] 未配置的模块使用全局日志级别

---

### 阶段5：文档和测试（优先级：P0）
**任务5.1**: 更新文档
- 在 `docs/` 目录添加日志系统文档
- 更新 `AGENT.md`

**任务5.2**: 单元测试
- 测试日志配置加载
- 测试 JSON 格式输出
- 测试环境变量覆盖

**任务5.3**: 集成测试
- 验证性能日志字段
- 验证错误日志字段

**验收标准**:
- [ ] 文档更新完成
- [ ] 测试覆盖率 > 80%
- [ ] 所有测试通过

---

## 四、文件变更清单

### 新增文件
```
backend/app/core/logging_config.py       # 日志配置模块
backend/app/core/logging_utils.py        # 日志工具函数
docs/LOGGING_GUIDE.md                    # 日志系统使用文档
backend/tests/unit/test_logging_config.py # 日志配置单元测试
```

### 修改文件
```
backend/main.py                          # 使用新的日志配置
backend/app/config.py                    # 添加日志相关配置项
backend/app/services/hybrid_chat_service.py  # 性能日志字段优化
backend/app/services/indexing_service.py     # 任务日志字段优化
backend/app/services/chat_service.py         # 错误日志字段优化
backend/pyproject.toml                   # 添加 python-json-logger 依赖
```

---

## 五、配置示例

### 环境变量
```bash
# 基础配置
export LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
export LOG_FORMAT=json                   # json 或 text
export LOG_OUTPUT=console                # console 或 file

# 文件日志配置（仅 LOG_OUTPUT=file 时有效）
export LOG_FILE_PATH=/var/log/orangerag/app.log
export LOG_FILE_MAX_BYTES=104857600      # 100MB
export LOG_FILE_BACKUP_COUNT=10

# 模块级别配置
export LOG_LEVEL_MODULE_OLLAMA=WARNING
export LOG_LEVEL_MODULE_RETRIEVERS=DEBUG
```

### JSON 日志输出示例
```json
{
  "timestamp": "2026-02-15T10:30:45.123Z",
  "level": "INFO",
  "module": "app.services.hybrid_chat_service",
  "message": "Hybrid retrieval completed",
  "service": "HybridChat",
  "duration_ms": 1250,
  "conversation_id": "abc-123",
  "stage": "retrieval"
}
```

```json
{
  "timestamp": "2026-02-15T10:30:45.456Z",
  "level": "ERROR",
  "module": "app.services.chat_service",
  "message": "Failed to generate response",
  "service": "Chat",
  "error_type": "TimeoutError",
  "error_message": "LLM request timed out after 180s",
  "stack_trace": "Traceback (most recent call last):...",
  "conversation_id": "abc-123"
}
```

---

## 六、回滚方案

如果出现问题，可通过环境变量快速回滚到文本格式：
```bash
export LOG_FORMAT=text
```

或在代码中注释掉 JSON 处理器配置。

---

## 七、进度跟踪

详见 todo 列表或在本节更新：

### 当前状态
- [x] 项目结构分析完成
- [x] 开发计划制定完成
- [x] 任务1.1: 日志配置文件 (backend/app/core/logging_config.py)
- [x] 任务1.2: 修改主入口 (backend/main.py)
- [x] 任务2.1: 性能字段 (log_performance 函数)
- [x] 任务2.2: 上下文字段 (conversation_id, task_id)
- [x] 任务2.3: 错误字段 (log_error 函数)
- [x] 任务3.1: 文件日志配置 (RotatingFileHandler)
- [x] 任务3.2: 生产环境配置 (文件路径、轮转)
- [x] 任务3.3: 开发环境配置 (控制台输出)
- [x] 任务4.1: 模块级别配置 (get_module_log_levels)
- [x] 任务4.2: 常用模块预配置 (环境变量支持)
- [x] 任务5.1: 文档更新 (docs/LOGGING_GUIDE.md)
- [x] 任务5.2: 单元测试 (backend/tests/unit/test_logging_config.py)
- [x] 任务5.3: 集成测试 (集成测试用例已包含)

### 完成度统计
- **阶段1 基础配置**: 100% (2/2 任务)
- **阶段2 增强字段**: 100% (3/3 任务)
- **阶段3 日志轮转**: 100% (3/3 任务)
- **阶段4 模块级别控制**: 100% (2/2 任务)
- **阶段5 文档和测试**: 100% (3/3 任务)
- **总体进度**: 100% (13/13 任务)

### 已实现功能
1. **JSON/Text 双格式支持**: 可通过 `LOG_FORMAT` 环境变量切换
2. **完整的字段支持**: timestamp, level, module, message, service, stage, duration_ms, conversation_id, task_id, error_type, error_message, stack_trace
3. **文件日志轮转**: 自动轮转，支持大小和备份数配置
4. **模块级别日志控制**: 通过 `LOG_LEVEL_MODULE_{NAME}` 环境变量
5. **辅助函数**: `log_performance()` 和 `log_error()` 便于记录性能和错误
6. **完整文档**: 338 行使用指南
7. **完整测试**: 346 行单元测试，覆盖率 > 80%

### 服务集成情况
- [x] backend/main.py - 已集成
- [x] backend/app/services/hybrid_chat_service.py - 已集成性能日志和错误日志
- [x] backend/app/services/indexing_service.py - 已集成结构化日志
- [x] backend/app/services/chat_service.py - 已集成结构化日志

---

*计划创建时间: 2026-02-15*  
*实际完成时间: 2026-02-15*  
*状态: ✅ 已完成*
