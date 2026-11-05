"""Planner：生成行程骨架（不含具体点位，规避幻觉）。

只输出「第几天 / 哪个区域 / 什么主题 / 时段框架 / 体力预算」，
具体点位由 Worker 检索后交给 Assembler 物化，从根本上切断开放式生成的幻觉路径。
"""
import os

from core.llm import extract_json_object, get_llm
from core.memory import get_memory_hub
from core.rules import AREAS

PLANNER_PROMPT = """你是家庭出行管家的行程骨架规划器。请输出**骨架**，不要指定具体景点或餐厅名称。

{memory}

用户请求：{query}
已抽取的硬约束：
{constraints}

可用区域：{areas}

只输出如下 JSON（不要解释、不要代码块标记）：
{{
  "days": 3,
  "skeleton": [
    {{"day": 1, "area": "南山区", "theme": "滨海平缓路线",
      "periods": ["上午", "下午"], "walk_budget_km": 3.0, "note": "首日安排轻松些"}}
  ]
}}

原则：
- 天数与 constraints.days 一致；
- 老人/儿童同行时每日景点不超过 2 个、walk_budget_km 不超过 3；
- 相邻两天尽量安排同一或相邻区域，减少跨区奔波；
- 每天至少留出一个时段作为弹性/休息。
"""


def _fast_mode() -> bool:
    return os.getenv("FAST_MODE", "0").strip() in ("1", "true", "True")


def deterministic_skeleton(constraints: dict) -> dict:
    """规则版骨架：按天轮换区域，节奏保守。"""
    days = int(constraints.get("days", 3) or 3)
    areas = constraints.get("areas") or []
    pool = [a for a in areas if a in AREAS] or AREAS[:days]

    walk = 3.0 if constraints.get("with_elderly") else 4.0
    skeleton = []
    for d in range(1, days + 1):
        area = pool[(d - 1) % len(pool)]
        skeleton.append({
            "day": d,
            "area": area,
            "theme": "平缓轻松路线" if constraints.get("with_elderly") or constraints.get("with_kids")
                     else "经典打卡路线",
            "periods": ["上午", "下午"],
            "walk_budget_km": walk,
            "note": "老人/儿童同行，节奏放缓",
        })
    return {"days": days, "skeleton": skeleton}


def planner_node(state: dict) -> dict:
    """Planner 节点：产出或复用行程骨架。"""
    constraints = state.get("constraints", {}) or {}
    sid = state.get("session_id", "default")
    hub = get_memory_hub()

    # replan 意图且已有骨架时复用，避免无谓重排；
    # 但若 Replanner 标记了「全量重规划」（局部重填多轮仍未通过），则必须重新生成
    existing = state.get("plan", {}) or {}
    if (state.get("intent") == "replan" and existing.get("skeleton")
            and not constraints.get("_full_replan")):
        return {"plan": existing}

    if _fast_mode():
        new_plan = deterministic_skeleton(constraints)
    else:
        try:
            prompt = PLANNER_PROMPT.format(
                memory=hub.inject(sid, "planner") or "（无历史记忆）",
                query=state.get("query", ""),
                constraints=constraints,
                areas="、".join(AREAS),
            )
            resp = get_llm().invoke([{"role": "user", "content": prompt}])
            parsed = extract_json_object(resp.content)
            skeleton = parsed.get("skeleton") or []
            days = int(parsed.get("days", constraints.get("days", 3)) or 3)
            if not skeleton:
                new_plan = deterministic_skeleton(constraints)
            else:
                new_plan = {"days": days, "skeleton": skeleton}
        except Exception:
            new_plan = deterministic_skeleton(constraints)

    # 消费全量重规划标记，避免后续轮次误判
    if constraints.get("_full_replan"):
        updated = dict(constraints)
        updated.pop("_full_replan", None)
        return {"plan": new_plan, "constraints": updated}
    return {"plan": new_plan}
