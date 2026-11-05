"""Replanner：Reflection 回跳，做「拦截 → 触发 Replan → 局部 State 更新」。

不重排整份行程，只针对 Critic 检出的违规做最小改动：
把违规实体加入排除集、收紧当日体力预算、按需强制室内优先，
随后回跳 assembler 局部重填；连续多轮仍不通过才升级为 planner 全量重规划。
"""
import os


def _max_replan() -> int:
    try:
        return max(1, int(os.getenv("MAX_REPLAN", "3")))
    except ValueError:
        return 3


def replanner_node(state: dict) -> dict:
    """Replanner 节点：局部更新 State，并决定下一跳。"""
    report = state.get("critic_report", {}) or {}
    constraints = dict(state.get("constraints", {}) or {})
    exclude = set(constraints.get("_exclude") or [])

    touched_codes = set()
    for v in report.get("violations", []):
        code = v.get("code", "")
        touched_codes.add(code)
        if v.get("severity") == "fatal" and v.get("ref_id"):
            exclude.add(v["ref_id"])

    # 体力类违规：收紧预算而不是整份推翻
    if "ELDER_STAMINA" in touched_codes or "TOO_MANY_POI" in touched_codes:
        constraints["max_attractions_per_day"] = max(
            1, int(constraints.get("max_attractions_per_day", 2) or 2) - 1)
        constraints["daily_walk_km_limit"] = round(
            float(constraints.get("daily_walk_km_limit", 3.0) or 3.0) * 0.8, 2)

    # 天气类违规：强制室内优先
    if "OUTDOOR_IN_RAIN" in touched_codes:
        constraints["avoid_outdoor_on_rain"] = True
        constraints["_prefer_indoor"] = True

    # 营业时间类违规：晚餐就近，避免跨区赶路
    if "CLOSED" in touched_codes:
        constraints["_strict_hours"] = True

    constraints["_exclude"] = sorted(exclude)

    replan_count = int(state.get("replan_count", 0) or 0) + 1
    next_hop = "assembler" if replan_count <= 2 else "planner"

    # 升级到全量重规划时打标记，让 Planner 真正重新生成骨架而非复用旧骨架
    if next_hop == "planner":
        constraints["_full_replan"] = True

    entry = {
        "round": replan_count,
        "fatal_count": report.get("fatal_count", 0),
        "codes": sorted(touched_codes),
        "excluded": sorted(exclude),
        "next": next_hop,
    }

    # 追加而非覆盖，保证回跳历史在 UI 中完整可见
    history = list(state.get("replan_history", []) or [])
    history.append(entry)

    return {
        "constraints": constraints,
        "replan_count": replan_count,
        "replan_history": history,
        "_next": next_hop,
    }


def route_after_replan(state: dict) -> str:
    """条件边：决定回跳到 assembler（局部重填）还是 planner（全量重规划）。"""
    return state.get("_next", "assembler")
