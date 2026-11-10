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
- 降级不是失败：即使 LLM 生成的 Cypher 完全不可用，也能通过多跳 GraphRAG / 向量检索给出可用召回。

### 1.1 多跳 GraphRAG 检索（`retrieval/graph_rag.py`）
在实体图上做两跳关系推理，并输出可解释的路径证据：
- 关系路径：`(:POI)-[:LOCATED_IN]->(:Area)`、`(:POI|:Restaurant)-[:NEAR]->(:Hotel)`、`(:POI)-[:LOCATED_IN]->(:Area)<-[:LOCATED_IN]-(:Restaurant)`；
- 混合排序：图路径权重 + 硬约束契合度 + 向量相似度；
- 每条结果附带 `graph_hops` / `graph_path` / `graph_score`，可在「检索链路」中查看推理依据；
- `area_summary()` 提供区域级社区摘要（数量 / 室内占比 / 平均强度 / 高频标签）；
- 快速模式默认走该路径，Text2Cypher 降级时也以其兜底——保证核心检索链路在无 LLM 时同样可运行、可评测。

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

知识库为「深圳家庭游」真实地点数据（`data/shenzhen.py`），共 **193 个实体**（107 个景点 / 74 家餐厅 / 12 家酒店），全部为深圳真实存在的地点，**不含程序化合成的占位实体**：

- 覆盖南山、福田、罗湖、盐田、龙岗、宝安、龙华、光明、坪山、大鹏新区 10 个区域；
- 每个景点带体力强度、是否室内、是否无障碍、门票、开放时间、地铁站、亲子/老人适配度与一句话提示；
- 餐厅以真实老字号与连锁品牌为主（点都德、陶陶居、八合里海记、润园四季、面点王、嘉旺……），带清淡指数、人均、卫生风险；
- 坐标取公开地图的街区级近似值，门票与营业时间为公开信息近似值，**出行前请以景区/商户当日公告为准**。

配图位于 `static/photos/`（每张图对应 `manifest.json` 中的标题与来源页）。图片由 `scripts/fetch_photos.py` 从公开图文检索中抓取，**只接受标题命中地点名的候选**，因此不会出现同名异地或无关配图；抓不到可靠实景图的地点会退回排版占位，不使用 AI 生成图。图片仅用于本地演示，正式发布前请替换为授权素材。

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
│   ├── graph_rag.py    # 多跳 GraphRAG（图遍历 + 向量混合排序 + 区域摘要）
│   ├── vector_store.py # 可插拔向量检索
│   ├── cache.py        # 可插拔缓存
│   └── text2cypher.py  # 四步 Fallback Pipeline
├── agents/
│   ├── router.py / planner.py / workers.py / assembler.py
│   ├── critic.py / replanner.py / supervisor.py
│   └── __init__.py
├── data/shenzhen.py    # 深圳真实地点知识库（193 实体）
├── static/photos/      # 实景配图（按实体 id 命名，附 manifest.json 来源记录）
├── scripts/
│   └── fetch_photos.py # 抓取/更新实景配图（含标题相关性过滤与裁切）
├── evals/run.py        # 可执行回归脚本（验证三大防御机制）
├── tests/              # pytest 测试套件（CI 使用）
├── legacy/             # 原始开源项目的遗留 agent（仅存档，不参与运行）
├── pytest.ini
├── travel_agent.py     # Streamlit 可视化控制台
├── requirements.txt
├── apik.txt            # OpenRouter API Key（优先读取，已在 .gitignore）
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
- 行程结果（每个点位配实景缩略图）、检索链路（Text2Cypher Fallback）、Critic 校验报告、Replan 回跳历史、三层记忆、架构 DAG 可视化
- 一键导出 Markdown 行程

地图（Leaflet / folium 渲染，瓦片为普通图片，不依赖 WebGL）：
- 底图为纯卫星影像（高德 `style=6`），不叠注记层，压低饱和与亮度作为背景，路线与点位才是主角；
- 点位与路线使用高德 GCJ-02 坐标，程序内置 WGS-84 → GCJ-02 纠偏，避免整体偏移数百米；
- 路线走 OSRM 真实道路折线；**同一天内按「段」配色**（同色系由浅到深表示先后顺序），每段带方向箭头；
- 「全部 / 第 N 天」切换避免多天路线挤在一起；点行程中的「地图」按钮可定位并高亮该点，同时自动切到对应天。

> 瓦片与路径服务均为公开演示接口（AutoNavi / OSRM demo），商用请自行申请授权或替换 `AMAP_SAT`、`AMAP_ANNO`、`_osrm_route`。

配图维护（可选，需要联网）：

```powershell
python scripts/fetch_photos.py            # 只补齐缺失的配图
python scripts/fetch_photos.py --report   # 查看覆盖情况
python scripts/fetch_photos.py --force    # 全部重抓
```

## 测试与回归

```powershell
pytest                 # 27 个用例，CI 同款
python evals/run.py    # 可读的回归脚本输出
```

`tests/` 覆盖：数据规模与 ID 唯一性、硬约束抽取、Text2Cypher 防御校验、Critic 硬约束、Replan 收敛、GraphRAG 多跳路径、端到端生成。CI 配置见 `.github/workflows/ci.yml`（Python 3.9 + pytest）。

## 常见问题

**未检测到 API Key？** 检查 `apik.txt` 或 `.env`，且 `apik.txt` 优先级高于同名环境变量。

**智能体增强模式很慢？** `deepseek-v4.1-flash-1017` 为推理型模型，单次请求可能数十秒到数分钟；日常体验与评测建议用「快速模式」（确定性、秒级，架构与防御机制完全一致，仅检索/规划改用规则实现）。

**想接真实 Neo4j / Milvus / Redis？** 设置对应环境变量（见上表）且网络可达即可，代码会自动切换，无需改业务逻辑。

## 开源协议

本项目为开源项目，详见 LICENSE 文件（如适用）。基于 Streamlit、LangGraph、LangChain 与 OpenRouter 构建，感谢开源社区。
