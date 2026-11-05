"""Router：意图识别 + 硬约束抽取。

输出结构化 HardConstraints，供后续所有 Agent 共享；
规则抽取作为基线，LLM 抽取在其之上做增强，两者冲突时以更严格者为准。
"""
import os

from core.llm import extract_json_object, get_llm
from core.memory import get_memory_hub
from core.rules import merge_constraints, rule_extract
from core.state import INTENT_ITINERARY, INTENT_QA, INTENT_REPLAN

ROUTER_PROMPT = """你是家庭出行管家的意图路由与约束抽取模块。请从用户请求中抽取结构化约束。

{memory}

用户请求：{query}

只输出如下 JSON（不要解释、不要代码块标记）：
{{
  "intent": "itinerary | replan | qa",
  "constraints": {{
    "with_elderly": true/false,
    "with_kids": true/false,
    "wheelchair": true/false,
    "days": 整数,
    "areas": ["南山区"],
    "dietary": ["清淡"],
    "must_avoid": ["海鲜排档"],
    "budget_per_meal": 整数,
    "near_hotel_km": 数字,
    "max_intensity": 1-5,
    "max_attractions_per_day": 整数,
    "daily_walk_km_limit": 数字
  }},
  "missing_slots": ["缺少的关键信息，如 酒店名称"],
  "clarifications": ["需要向用户确认的一句话问题"]
}}

判断规则：
- itinerary=规划新行程；replan=已有行程需要调整；qa=仅就行程提问
- 提到老人/长辈即 with_elderly=true；提到孩子/带娃即 with_kids=true
- 提到轮椅/无障碍/婴儿车即 wheelchair=true
- 未明确说明的字段请按家庭游保守值推断，不要留空
"""


def _fast_mode() -> bool:
    return os.getenv("FAST_MODE", "0").strip() in ("1", "true", "True")


def router_node(state: dict) -> dict:
    """Router 节点：识别意图并抽取硬约束。"""
    query = state.get("query", "")
    sid = state.get("session_id", "default")
    hub = get_memory_hub()

    base = rule_extract(query)

    if _fast_mode():
        constraints = base
        intent = INTENT_REPLAN if state.get("plan") else INTENT_ITINERARY
        missing, clarifications = _missing_slots(base, query)
    else:
        try:
            prompt = ROUTER_PROMPT.format(
                memory=hub.inject(sid, "router") or "（无历史记忆）",
                query=query,
            )
            resp = get_llm().invoke([{"role": "user", "content": prompt}])
            parsed = extract_json_object(resp.content)
            intent = parsed.get("intent", INTENT_ITINERARY)
            if intent not in (INTENT_ITINERARY, INTENT_REPLAN, INTENT_QA):
                intent = INTENT_ITINERARY
            constraints = merge_constraints(base, parsed.get("constraints", {}))
            # LLM 可收紧安全阈值（更小），但不可放宽；距离偏好可放宽（更大），但不可收紧
            for safe in ("daily_walk_km_limit", "max_intensity", "max_attractions_per_day"):
                if safe in constraints:
                    constraints[safe] = min(base.get(safe, constraints[safe]), constraints[safe])
            for pref in ("near_hotel_km",):
                if pref in constraints:
                    constraints[pref] = max(base.get(pref, constraints[pref]), constraints[pref])
            missing = parsed.get("missing_slots", []) or []
            clarifications = parsed.get("clarifications", []) or []
        except Exception as e:
            constraints = base
            intent = INTENT_ITINERARY
            missing, clarifications = _missing_slots(base, query)
            clarifications = clarifications + [f"（约束抽取降级为规则模式：{type(e).__name__}）"]

    # 严格化：只要同行有老人，体力上限取两者更严
    if constraints.get("with_elderly"):
        constraints["max_intensity"] = min(int(constraints.get("max_intensity", 3)), 3)
        constraints["max_attractions_per_day"] = min(
            int(constraints.get("max_attractions_per_day", 2)), 2)

    # 写回用户层记忆
    hub.update_user(
        sid,
        elderly_count=2 if constraints.get("with_elderly") else 0,
        kid_ages=[6] if constraints.get("with_kids") else [],
        mobility="轮椅" if constraints.get("wheelchair") else "",
        dietary=constraints.get("dietary") or [],
    )

    return {
        "intent": intent,
        "constraints": constraints,
        "missing_slots": missing,
        "clarifications": clarifications,
        "user_profile": hub.user(sid).to_dict(),
    }


def _missing_slots(constraints: dict, query: str) -> tuple[list, list]:
    """规则模式下识别缺失信息。"""
    clarifications = []
    if constraints.get("_hotel_defaulted"):
        clarifications.append(
            f"未指定酒店，已默认按「{constraints.get('hotel_name', '')}」规划，如需调整请告知")
    if not constraints.get("days"):
        clarifications.append("请补充：行程天数")
    return [], clarifications
