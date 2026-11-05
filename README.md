# 家庭出行管家 · Supervisor-Worker 多智能体系统（MultiAgents with LangGraph）

一个**防御优先（defense-first）**的家庭出行行程规划多智能体系统。基于 Streamlit + LangGraph + OpenRouter 构建，针对家庭游场景（老人/儿童/轮椅/清淡饮食/天气突变）中的真实痛点，用一套可插拔检索层 + 确定性物化 + 硬约束回跳校验，系统性解决「**检索不可控、幻觉不可控、不可自我纠错**」三大问题。

> 本仓库在原始项目（[MultiAgents-with-Langgraph-TravelItineraryPlanner](https://github.com/vikrambhat2/MultiAgents-with-Langgraph-TravelItineraryPlanner)）基础上，按 `about.txt` 的设计目标重构了智能体架构与存储层，并将界面与提示词全面中文化、接入 OpenRouter。

---

## 核心架构（四阶段状态图）

```
        ┌─────────┐
   query→│ Router  │  意图识别 + 硬约束抽取（规则基线 + LLM 增强）
        └────┬────┘
             ↓
        ┌─────────┐
        │ Planner │  只生成「骨架」（哪天/哪个区/什么主题），规避开放式幻觉
        └────┬────┘
             ↓
   ┌─────────┴──────────┐  asyncio.gather 三路并发
   │  并发 Workers        │  ├ 图谱检索（Text2Cypher）
   │                     │  ├ 向量检索（降级兜底）
   │                     │  └ 天气 API
   └─────────┬──────────┘
             ↓
        ┌──────────┐
        │ Assembler│  确定性物化：从真实召回候选中筛选排布，杜绝幻觉点位
        └────┬─────┘
             ↓
        ┌─────────┐        fatal 且未达轮次上限
        │ Critic  │──────────────────────┐
        └────┬────┘                       ↓
             │ passed                    ┌─────────┐  拦截→触发→局部 State 更新
             ↓                          │ Replan  │──┐
        ┌─────────┐                     └─────────┘  │ 局部重填(assembler)
        │ Finalize│                                  └──┘ 或全量重规划(planner)
        └─────────┘
```

---

## 三大防御机制（对应 `about.txt` 的痛点）

### 1. 检索不可控 → Text2Cypher 防御性纠错链路
知识图谱查询走「**生成 → AST/Schema 校验 → LLM 重写 → 向量降级**」四步 Fallback Pipeline：
- Schema 白名单强制校验：**任何不在白名单的标签 / 关系 / 属性 / 语法都会被拦截并触发重写或直接降级**，从源头杜绝模型臆造图谱结构；
- 任意一步失败都被完整记录在 `retrieval_trace` 中，形成可观测链路（便于 LangSmith / 评测分析瓶颈）；
- 降级不是失败：即使 LLM 生成的 Cypher 完全不可用，也能通过向量检索给出可用召回。

### 2. 幻觉不可控 → Planner 骨架 + Assembler 确定性物化
- Planner **只输出行程骨架**（天数/区域/主题/时段框架/体力预算），不指定任何具体点位；
- Assembler 是**纯确定性**逻辑，只从 Worker 真实召回的实体里做筛选与排布，行程中不可能出现知识库外的幻觉点位。

### 3. 不可自我纠错 → Critic 硬约束校验 + Replan 回跳
- Critic 采用「**规则校验为主、LLM 语义复核为辅**」：老人体力阈值 / 无障碍 / 营业时间 / 天气预警 / 卫生风险等硬指标零容错；
- 校验失败时进入 **Replan**：做最小改动（把违规实体加入排除集、收紧当日体力预算、按需强制室内优先），再回跳 Assembler 局部重填；连续多轮仍不通过才升级为 Planner 全量重规划；达到轮次上限则带警告收尾。

### 额外：三层 Memory（Hook 注入）
会话层 / 行程层 / 用户层记忆，通过 `inject(scope)` 只向对应 Agent 注入所需上下文，避免全量历史灌入造成意图漂移。

---

## 可插拔存储层

三类基础设施均为「**真实服务优先、进程内回退**」，未配置或连不通时自动降级为零依赖实现，保证开箱即用：

| 层 | 真实后端（配置即启用） | 回退实现 |
|---|---|---|
| 图存储 | `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | 进程内图 + 受控 Cypher 子集解析器 |
| 向量 | `MILVUS_URI` | 字符二元组 TF-IDF 余弦（无需 embedding） |
| 缓存 | `REDIS_URL` | 进程内 TTL 缓存 |

种子知识库为「深圳家庭游」示例数据（`data/shenzhen.py`），接入真实业务时替换为权威数据源即可。

---

## 目录结构

```
MultiAgents-with-Langgraph-TravelItineraryPlanner/
├── core/
│   ├── state.py        # 全局 State Schema（TypedDict + 归约器，Agent 间消息隔离）
│   ├── geo.py          # 球面距离
│   ├── llm.py          # OpenRouter 统一 LLM 配置
│   ├── memory.py       # 三层 Memory + Hook 注入
│   └── rules.py        # 硬约束抽取 + 确定性校验规则
├── retrieval/
│   ├── schema.py       # 图谱 Schema（Text2Cypher 校验依据）
│   ├── graph_store.py  # 可插拔图存储 + 受控 Cypher 解析器
│   ├── vector_store.py # 可插拔向量检索
│   ├── cache.py        # 可插拔缓存
│   └── text2cypher.py  # 四步 Fallback Pipeline
├── agents/
│   ├── router.py / planner.py / workers.py / assembler.py
│   ├── critic.py / replanner.py / supervisor.py
│   └── __init__.py
├── data/shenzhen.py    # 深圳种子知识库（示例数据）
├── evals/run.py        # 回归评测集（验证三大防御机制）
├── travel_agent.py     # Streamlit 可视化控制台
├── requirements.txt
├── apik.txt            # OpenRouter API Key（优先读取）
└── .env
```

---

## 环境要求

- Python 3.9+
- 一个 OpenRouter API Key（[https://openrouter.ai/keys](https://openrouter.ai/keys)）

## 安装与配置

1. 创建并激活虚拟环境：`python -m venv .venv` → `.\.venv\Scripts\Activate.ps1`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置 Key（任选其一）：
   - 将 Key 字符串写入项目根目录 `apik.txt`（**优先级最高**）
   - 或在 `.env` 中设置 `OPENROUTER_API_KEY`
4. 设置模型（`.env`）：`OPENROUTER_MODEL=deepseek/deepseek-v4.1-flash-1017`（可替换为任意 OpenRouter 模型）

> 部分模型存在地区限制（返回 `403 This model is not available in your region`），更换模型即可。

## 运行

```powershell
streamlit run travel_agent.py
```

控制台提供：
- **两种模式**：⚡ 快速模式（确定性，秒级）/ 🤖 智能体增强（LLM，较慢但更智能）
- 行程结果、检索链路（Text2Cypher Fallback）、Critic 校验报告、Replan 回跳历史、三层记忆、架构 DAG 可视化
- 一键导出 Markdown 行程

## 回归评测

```powershell
python evals/run.py
```

评测集在 FAST_MODE 下运行，覆盖：硬约束抽取、Text2Cypher 防御校验、Critic 硬约束校验、Replan 回跳收敛、端到端生成，全绿即通过。

## 常见问题

**未检测到 API Key？** 检查 `apik.txt` 或 `.env`，且 `apik.txt` 优先级高于同名环境变量。

**智能体增强模式很慢？** `deepseek-v4.1-flash-1017` 为推理型模型，单次请求可能数十秒到数分钟；日常体验与评测建议用「快速模式」（确定性、秒级，架构与防御机制完全一致，仅检索/规划改用规则实现）。

**想接真实 Neo4j / Milvus / Redis？** 设置对应环境变量（见上表）且网络可达即可，代码会自动切换，无需改业务逻辑。

## 开源协议

本项目为开源项目，详见 LICENSE 文件（如适用）。基于 Streamlit、LangGraph、LangChain 与 OpenRouter 构建，感谢开源社区。
