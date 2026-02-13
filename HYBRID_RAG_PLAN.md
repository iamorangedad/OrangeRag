# Hybrid RAG 开发计划文档

**项目**: Smart Document Assistant - Hybrid RAG 扩展  
**创建日期**: 2026-02-07  
**最后更新**: 2026-02-07  
**版本**: v1.0  

---

## 1. 项目背景与目标

### 1.1 当前RAG方案

本项目当前采用**密集检索（Dense Retrieval）**架构，基于：
- **嵌入模型**: `nomic-embed-text` (768维)
- **向量存储**: Simple内存存储 / ChromaDB持久化
- **检索算法**: 余弦相似度
- **LLM**: Ollama (`qwen3:4b`)

### 1.2 当前局限

| 问题 | 影响 |
|------|------|
| 单一检索方式 | 仅依赖向量检索，对精确关键词匹配能力弱 |
| 缺乏重排序 | 直接取Top-K，未考虑多路召回融合 |
| 无查询扩展 | 原始查询直接用于检索 |
| 同义词/缩写处理差 | 向量检索对专业术语的同义词表现不佳 |

### 1.3 Hybrid RAG目标

通过引入**稀疏检索（Sparse Retrieval）**和**结果融合**，实现：
- 语义匹配 + 精确关键词匹配的双通道检索
- RRF融合算法提升召回率和准确性
- 可选的重排序层进一步优化结果

---

## 2. 目标架构设计

```
用户查询
    │
    ▼
查询预处理器 (Phase 3 - 可选)
    │
    ├─────────────┬─────────────┐
    ▼             │             ▼
密集检索通道      │      稀疏检索通道
(Dense)           │       (Sparse/BM25)
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
         RRF结果融合层
                  │
                  ▼
         重排序层 (Phase 3 - 可选)
                  │
                  ▼
            LLM生成回答
```

### 2.1 技术选型

| 组件 | 技术选择 | 理由 |
|------|----------|------|
| 密集检索 | 现有向量存储 | 保持兼容性 |
| 稀疏检索 | BM25 (rank-bm25库) | 经典算法，效果好 |
| 融合算法 | RRF (Reciprocal Rank Fusion) | 无需训练，对排序敏感 |
| 重排序 | Cross-Encoder (Sentence-Transformers) | Phase 3实现 |

---

## 3. 模块设计

### 3.1 目录结构

```
backend/app/
├── core/
│   ├── retrievers/                    # 新增: 检索器模块
│   │   ├── __init__.py
│   │   ├── base.py                    # 检索器抽象基类
│   │   ├── dense_retriever.py         # 密集检索器
│   │   ├── sparse_retriever.py        # 稀疏检索器 (BM25)
│   │   └── hybrid_retriever.py        # 混合检索器 (RRF融合)
│   ├── fusion/                        # 新增: 融合算法
│   │   ├── __init__.py
│   │   ├── rrf_fusion.py              # RRF融合实现
│   │   └── weighted_fusion.py         # 加权融合
│   └── reranker/                      # 新增: 重排序模块 (Phase 3)
│       ├── __init__.py
│       ├── base.py
│       └── cross_encoder.py           # 交叉编码器重排序
├── services/
│   ├── chat_service.py                # 现有服务
│   └── hybrid_chat_service.py         # 新增: Hybrid RAG聊天服务
└── config.py                          # 新增: Hybrid配置项
```

### 3.2 关键接口设计

```python
# 检索器基类接口
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """检索文档节点"""
        pass

# 融合算法接口
class FusionAlgorithm(ABC):
    @abstractmethod
    def fuse(self, results: List[List[NodeWithScore]]) -> List[NodeWithScore]:
        """融合多路检索结果"""
        pass
```

---

## 4. 实施路线图

### Phase 1: 基础Hybrid实现 (优先级: 高)

| 任务 | 状态 | 负责人 | 预计工时 | 依赖 |
|------|------|--------|----------|------|
| 1.1 添加rank-bm25依赖 | 已完成 | - | 0.5h | - |
| 1.2 创建检索器基类 | 已完成 | - | 2h | 1.1 |
| 1.3 实现DenseRetriever封装 | 已完成 | - | 3h | 1.2 |
| 1.4 实现BM25稀疏检索器 | 已完成 | - | 4h | 1.2 |
| 1.5 实现RRF融合算法 | 已完成 | - | 3h | - |
| 1.6 创建HybridRetriever整合 | 已完成 | - | 4h | 1.3-1.5 |
| 1.7 更新配置类添加Hybrid配置 | 已完成 | - | 2h | - |
| 1.8 创建HybridChatService | 已完成 | - | 4h | 1.6-1.7 |
| 1.9 添加单元测试 | 已完成 | - | 4h | 全部 |
| 1.10 集成测试与调优 | 已完成 | - | 4h | 1.9 |

**Phase 1 交付物:**
- 可用的Hybrid RAG检索流程
- 可配置的密集/稀疏权重
- 基础测试覆盖

### Phase 2: 性能优化 (优先级: 中)

| 任务 | 状态 | 负责人 | 预计工时 | 依赖 |
|------|------|--------|----------|------|
| 2.1 BM25索引缓存机制 | 已完成 | - | 3h | Phase 1 |
| 2.2 异步并行检索实现 | 已完成 | - | 4h | Phase 1 |
| 2.3 加权融合算法支持 | 已完成 | - | 2h | Phase 1 |
| 2.4 性能基准测试 | 已完成 | - | 3h | 2.1-2.3 |
| 2.5 配置调优指南 | 已完成 | - | 2h | 2.4 |

**Phase 2 交付物:**
- 索引缓存提升加载速度
- 并行检索减少延迟
- 性能测试报告

### Phase 3: 高级功能 (优先级: 低)

| 任务 | 状态 | 负责人 | 预计工时 | 依赖 |
|------|------|--------|----------|------|
| 3.1 Cross-Encoder重排序器 | 已完成 | - | 6h | Phase 2 |
| 3.2 查询扩展/重写模块 | 已完成 | - | 8h | Phase 2 |
| 3.3 A/B测试框架 | 已完成 | - | 6h | 全部 |
| 3.4 完整文档更新 | 已完成 | - | 4h | 全部 |

**Phase 3 交付物:**
- 可选的重排序层
- 查询增强功能
- 效果评估工具

---

## 5. 技术实现规范

### 5.1 依赖项

新增依赖 (添加到 pyproject.toml):

```toml
[project.dependencies]
# ... 现有依赖 ...
rank-bm25 = ">=0.2.2"           # BM25实现
```

Phase 3 可选依赖:
```toml
sentence-transformers = ">=2.2.0"  # Cross-Encoder重排序
```

### 5.2 配置规范

新增配置项 (backend/app/config.py):

```python
# Hybrid RAG Configuration
enable_hybrid_search: bool = Field(default=True, alias="ENABLE_HYBRID_SEARCH")
dense_weight: float = Field(default=0.5, alias="DENSE_WEIGHT")
sparse_weight: float = Field(default=0.5, alias="SPARSE_WEIGHT")
rrf_k: float = Field(default=60.0, alias="RRF_K")

# 检索配置
dense_top_k: int = Field(default=10, alias="DENSE_TOP_K")
sparse_top_k: int = Field(default=10, alias="SPARSE_TOP_K")
final_top_k: int = Field(default=5, alias="FINAL_TOP_K")

# 重排序配置 (Phase 3)
enable_rerank: bool = Field(default=False, alias="ENABLE_RERANK")
rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
```

### 5.3 RRF算法规范

公式:
```
RRF_Score(d) = Σ 1/(k + rank_i(d))

其中:
- d: 文档
- k: 常数 (默认60，论文推荐值)
- rank_i(d): 文档d在第i个检索结果中的排名
```

代码实现要求:
- 处理文档ID冲突
- 支持任意数量的检索通道
- 时间复杂度: O(n log n)，n为文档总数

---

## 6. 测试策略

### 6.1 单元测试

| 测试模块 | 覆盖内容 | 数量 |
|----------|----------|------|
| sparse_retriever | BM25索引构建、检索 | >= 5 |
| rrf_fusion | RRF分数计算、边界情况 | >= 5 |
| hybrid_retriever | 双通道集成、配置切换 | >= 5 |

### 6.2 集成测试

- 端到端Hybrid RAG流程
- 不同配置组合测试
- 性能基准测试

### 6.3 评估指标

| 指标 | Dense基线 | Hybrid目标 | 测试方法 |
|------|-----------|------------|----------|
| Recall@10 | 基线值 | +15-25% | 标准测试集 |
| MRR | 基线值 | +10-20% | 人工评估 |
| 平均延迟 | 基线值 | +<50ms | 压力测试 |

---

## 7. 状态追踪

### 7.1 任务总览

| Phase | 总任务数 | 已完成 | 进行中 | 未开始 | 进度 |
|-------|----------|--------|--------|--------|------|
| Phase 1 | 10 | 10 | 0 | 0 | 100% |
| Phase 2 | 5 | 5 | 0 | 0 | 100% |
| Phase 3 | 4 | 4 | 0 | 0 | 100% |
| **总计** | **19** | **19** | **0** | **0** | **100%** |

### 7.2 最近更新

| 日期 | 更新内容 | 完成模块 |
|------|----------|----------|
| 2026-02-07 | 创建开发计划文档 | - |
| 2026-02-07 | 完成Phase 1全部任务 | 1.1-1.10 |
| 2026-02-07 | 完成Phase 2全部任务 | 2.1-2.5 |
| 2026-02-07 | 完成Phase 3全部任务 | 3.1-3.4 |
| - | - | - |

### 7.3 下一步行动

当前阶段: **全部完成**

状态: **所有Phase已完成，Hybrid RAG系统开发完毕**

---

## 8. 附录

### 8.1 参考资源

- [RRF论文: Reciprocal Rank Fusion outperforms Condorcet](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BM25算法详解](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- [LlamaIndex检索器文档](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/)

### 8.2 术语表

| 术语 | 解释 |
|------|------|
| Dense Retrieval | 基于向量嵌入的语义检索 |
| Sparse Retrieval | 基于关键词的统计检索 (如BM25) |
| RRF | Reciprocal Rank Fusion，倒数排名融合算法 |
| Cross-Encoder | 交叉编码器，用于重排序的神经网络模型 |
| Hybrid RAG | 混合检索增强生成，结合多种检索方式 |

---

## 9. 变更日志

### v1.0 (2026-02-07)
- 初始版本创建
- 定义Phase 1-3实施路线
- 确定技术选型

