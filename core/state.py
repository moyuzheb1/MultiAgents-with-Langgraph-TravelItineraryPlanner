"""全局 State Schema。

用 TypedDict 定义单一事实来源，所有 Agent 只读自己关心的字段、
只写自己负责的字段，从而实现 Agent 间的消息隔离与上下文精准注入。
带 Annotated 归约器的字段（warnings / retrieval_trace）由多个 Agent 追加写入。
"""
import operator
from typing import Annotated, Any, TypedDict

# ---------------- 意图类型 ----------------
INTENT_ITINERARY = "itinerary"      # 生成完整行程
INTENT_REPLAN = "replan"            # 行中重规划（突发天气/临时变更）
INTENT_QA = "qa"                    # 行程问答


class HardConstraints(TypedDict, total=False):
    """从 Query 中抽取的硬约束，Critic 会逐条校验。"""
    with_elderly: bool               # 同行有老人
    with_kids: bool                  # 同行有小孩
    wheelchair: bool                 # 需要无障碍通道
    daily_walk_km_limit: float       # 老人单日步行上限（公里）
    max_attractions_per_day: int     # 单日景点上限
    max_intensity: int               # 单景点体力强度上限（1-5）
    dietary: list[str]               # 饮食要求：清淡 / 不吃海鲜 / 素食
    avoid_outdoor_on_rain: bool      # 雨天避免室外
    near_hotel_km: float             # 餐厅距酒店上限（公里）
    budget_per_meal: int             # 人均餐标
    areas: list[str]                 # 限定区域
    days: int                        # 行程天数
    must_avoid: list[str]            # 明确规避项（如"海鲜排档"）


class PlanSlot(TypedDict, total=False):
    """行程中的一个时间槽。"""
    day: int
    period: str                      # 上午 / 下午 / 晚上
    kind: str                        # poi / dining / rest
    ref_id: str                      # 指向知识库实体
    name: str
    note: str


class Violation(TypedDict, total=False):
    """Critic 检出的一条违规。"""
    code: str                        # 违规码，如 ELDER_STAMINA / OUTDOOR_IN_RAIN
    severity: str                    # fatal / warning
    day: int
    ref_id: str
    detail: str
    suggestion: str


class WorkerPayload(TypedDict, total=False):
    """单个 Worker 的返回结果。"""
    source: str                      # graph / vector / api
    status: str                      # ok / degraded / error
    latency_ms: int
    data: Any
    error: str
    attempts: list                   # Text2Cypher 的 Fallback 链路轨迹


class TravelState(TypedDict, total=False):
    # ---- 输入 ----
    query: str
    session_id: str

    # ---- Router ----
    intent: str
    constraints: HardConstraints
    missing_slots: list[str]
    clarifications: list[str]

    # ---- Memory 注入 ----
    user_profile: dict
    itinerary_memory: dict
    session_summary: str

    # ---- Planner ----
    plan: dict                       # {"days": int, "slots": [PlanSlot]}

    # ---- Workers（asyncio.gather 三路并发） ----
    worker_results: dict             # {"poi": WorkerPayload, "dining": ..., "weather": ...}
    retrieval_trace: Annotated[list[dict], operator.add]

    # ---- Critic / Reflection ----
    critic_report: dict              # {"passed": bool, "violations": [Violation], "fatal_count": int}
    replan_count: int
    replan_history: Annotated[list[dict], operator.add]
    _next: str                       # Replanner 决定的下一跳（assembler / planner）

    # ---- 输出 ----
    answer: str
    warnings: Annotated[list[str], operator.add]
    final: bool


def initial_state(query: str, session_id: str = "default") -> TravelState:
    """构造一次请求的初始状态。"""
    return TravelState(
        query=query,
        session_id=session_id,
        intent=INTENT_ITINERARY,
        constraints=HardConstraints(),
        missing_slots=[],
        clarifications=[],
        user_profile={},
        itinerary_memory={},
        session_summary="",
        plan={},
        worker_results={},
        retrieval_trace=[],
        critic_report={},
        replan_count=0,
        replan_history=[],
        _next="",
        answer="",
        warnings=[],
        final=False,
    )
