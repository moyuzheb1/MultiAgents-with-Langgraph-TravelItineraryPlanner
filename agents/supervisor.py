"""Supervisor：LangGraph 状态图编排（Router → Planner → 并发 Workers → Assembler → Critic）。

条件边：
    critic --通过--> finalize --> END
    critic --致命违规且未达轮次上限--> replan --局部--> assembler --> critic   (回跳)
                                            └------全量--> planner  --> ...

其中 replan → assembler 为「局部 State 更新」回跳，replan → planner 为升级后的全量重规划。

`stream_trip()` 以 stream_mode="updates" 逐节点产出状态快照，供界面展示中间过程。
"""
import os
import time

from langgraph.graph import END, StateGraph

from agents.assembler import assembler_node
from agents.critic import critic_node
from agents.planner import planner_node
from agents.replanner import replanner_node, route_after_replan
from agents.router import router_node
from agents.workers import workers_node
from core.memory import Turn, get_memory_hub
from core.state import TravelState, initial_state


def _max_replan() -> int:
    try:
        return max(1, int(os.getenv("MAX_REPLAN", "3")))
    except ValueError:
        return 3


def finalize_node(state: dict) -> dict:
    """收尾节点：渲染最终行程 Markdown，并把结果写回三层 Memory。"""
    plan = state.get("plan", {}) or {}
    report = state.get("critic_report", {}) or {}
    weather = (state.get("worker_results", {}).get("weather") or {}).get("data") or {}
    constraints = state.get("constraints", {}) or {}
    hotel = constraints.get("hotel_name", "")
    sid = state.get("session_id", "default")
    hub = get_memory_hub()

    days = int(plan.get("days") or constraints.get("days") or 3)
    slots = plan.get("slots", []) or []

    lines = [f"## 深圳家庭游行程（{days} 天）", ""]
    if hotel:
        lines.append(f"**参考酒店**：{hotel}")
    if constraints.get("with_elderly"):
        lines.append(f"**老人体力预算**：每日 ≤{constraints.get('daily_walk_km_limit')} km，"
                     f"单景点强度 ≤{constraints.get('max_intensity')}")
    lines.append("")

    for d in range(1, days + 1):
        w = weather.get(str(d), {})
        head = f"### 第 {d} 天"
        if w:
            head += f" · {w.get('level', '')} {w.get('temp', '')}"
        lines.append(head)
        day_slots = [s for s in slots if s.get("day") == d]
        if not day_slots:
            lines.append("- （当日无满足约束的安排，建议改为酒店休息或自由活动）")
        for s in day_slots:
            icon = {"poi": "📍", "dining": "🍽️"}.get(s.get("kind"), "•")
            lines.append(f"- {icon} **{s.get('period')}** {s.get('name')} — {s.get('note', '')}")
        if w and w.get("rain"):
            lines.append(f"  - ⚠️ 天气提示：{w.get('advice', '')}")
        lines.append("")

    violations = report.get("violations", []) or []
    fatal = [v for v in violations if v.get("severity") == "fatal"]
    warns = [v for v in violations if v.get("severity") != "fatal"]

    if fatal:
        lines.append("### ⛔ 仍未消除的致命违规")
        for v in fatal:
            lines.append(f"- 第{v.get('day')}天 [{v.get('code')}] {v.get('detail')} → {v.get('suggestion')}")
        lines.append("")

    if warns:
        lines.append("### ⚠️ 提示")
        for v in warns[:8]:
            lines.append(f"- 第{v.get('day')}天 [{v.get('code')}] {v.get('detail')}")
        lines.append("")

    if report.get("passed"):
        lines.append(f"✅ 全部硬约束校验通过（重规划 {state.get('replan_count', 0)} 轮）")
    if state.get("clarifications"):
        lines.append("**需确认**：" + "；".join(state["clarifications"]))

    answer = "\n".join(lines)

    # 写回三层记忆
    hub.record_turn(sid, Turn(
        query=state.get("query", ""),
        intent=state.get("intent", ""),
        answer=answer,
        summary=f"{days}天行程/{len(slots)}个安排/致命违规{len(fatal)}条",
    ))
    hub.update_itinerary(sid, plan, reason=f"校验{'通过' if report.get('passed') else '未通过'}")

    return {"answer": answer, "final": True}


def route_after_critic(state: dict) -> str:
    """条件边：通过则收尾，否则回跳重规划（达上限则带警告收尾）。"""
    report = state.get("critic_report", {}) or {}
    if report.get("passed"):
        return "finalize"
    if int(state.get("replan_count", 0) or 0) >= _max_replan():
        return "finalize"
    return "replan"


def build_graph():
    workflow = StateGraph(TravelState)

    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("workers", workers_node)
    workflow.add_node("assembler", assembler_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("replan", replanner_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "planner")
    workflow.add_edge("planner", "workers")
    workflow.add_edge("workers", "assembler")
    workflow.add_edge("assembler", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {"finalize": "finalize", "replan": "replan"},
    )
    workflow.add_conditional_edges(
        "replan",
        route_after_replan,
        {"assembler": "assembler", "planner": "planner"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


GRAPH = None


def get_graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = build_graph()
    return GRAPH


def run_trip(query: str, session_id: str = "default") -> dict:
    """执行一次完整的 Supervisor-Worker 流程。"""
    return get_graph().invoke(initial_state(query, session_id))


# 带 Annotated[..., operator.add] 归约器的字段，多节点写入时按追加合并
_ADDITIVE_FIELDS = ("retrieval_trace", "replan_history", "warnings")


def _merge_update(state: dict, update: dict) -> dict:
    """把单节点产出的增量合并进状态快照，语义与图的归约器保持一致。"""
    merged = dict(state)
    for key, value in (update or {}).items():
        if key in _ADDITIVE_FIELDS and isinstance(value, list):
            merged[key] = list(merged.get(key) or []) + value
        else:
            merged[key] = value
    return merged


def stream_trip(query: str, session_id: str = "default"):
    """逐节点执行，每完成一个节点 yield (节点名, 状态快照, 累计耗时秒)。

    与 run_trip 走同一张图、同一套逻辑，仅把整段阻塞换成节点级流式，
    供界面实时展示 router / planner / workers / critic / replan 的推进过程。
    """
    state = initial_state(query, session_id)
    started = time.time()
    for chunk in get_graph().stream(state, stream_mode="updates"):
        for node, update in chunk.items():
            state = _merge_update(state, update)
            yield node, state, time.time() - started
