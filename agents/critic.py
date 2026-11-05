"""Critic：硬约束校验 + Reflection。

采用「规则校验为主、LLM 语义复核为辅」的双层结构：
- 规则层：体力阈值 / 无障碍 / 营业时间 / 天气预警 / 卫生风险等硬指标，确定性、零容错
- LLM 层：对规则无法覆盖的语义问题（如节奏失衡、跨区奔波）做补充复核
规则层的 fatal 违规直接决定是否需要回跳重规划。
"""
import os

from core.llm import extract_json_object, get_llm
from core.memory import get_memory_hub
from core.rules import validate_plan

CRITIC_PROMPT = """你是家庭出行的安全与体验复核官。下面是一份行程与规则引擎已检出的违规项，
请只对**规则未覆盖的语义问题**做补充复核。

{memory}

硬约束：{constraints}

行程：
{plan}

规则引擎已检出：
{rule_violations}

只输出如下 JSON（不要解释、不要代码块标记）：
{{
  "extra_violations": [
    {{"code": "SEMANTIC", "severity": "warning", "day": 1, "ref_id": "",
      "detail": "问题描述", "suggestion": "改进建议"}}
  ],
  "summary": "一句话总体评价"
}}

原则：
- 只补充真正影响老人/儿童体验的问题，最多 3 条；
- 不要重复规则引擎已列出的违规；
- 没有问题就返回空数组。
"""


def _fast_mode() -> bool:
    return os.getenv("FAST_MODE", "0").strip() in ("1", "true", "True")


def _llm_critic(state, rule_violations: list) -> list:
    """LLM 语义复核（可通过 ENABLE_LLM_CRITIC=0 关闭）。"""
    if os.getenv("ENABLE_LLM_CRITIC", "1").strip() not in ("1", "true", "True"):
        return []
    sid = state.get("session_id", "default")
    hub = get_memory_hub()

    plan = state.get("plan", {}) or {}
    lines = []
    for s in plan.get("slots", []):
        lines.append(f"第{s.get('day')}天 {s.get('period')} [{s.get('kind')}] {s.get('name')} "
                     f"({s.get('note', '')})")

    try:
        prompt = CRITIC_PROMPT.format(
            memory=hub.inject(sid, "critic") or "（无历史记忆）",
            constraints=state.get("constraints", {}),
            plan="\n".join(lines) or "（空行程）",
            rule_violations=rule_violations or "无",
        )
        resp = get_llm().invoke([{"role": "user", "content": prompt}])
        parsed = extract_json_object(resp.content)
        extra = parsed.get("extra_violations") or []
        return [v for v in extra if isinstance(v, dict)][:3]
    except Exception:
        return []


def critic_node(state: dict) -> dict:
    """Critic 节点：规则校验 + 可选 LLM 复核，产出 critic_report。"""
    constraints = state.get("constraints", {}) or {}
    plan = state.get("plan", {}) or {}
    weather = (state.get("worker_results", {}).get("weather") or {}).get("data") or {}
    hotel_name = constraints.get("hotel_name", "")

    violations = validate_plan(plan, constraints, weather, hotel_name)

    if not _fast_mode():
        violations = violations + _llm_critic(state, violations)

    fatal = [v for v in violations if v.get("severity") == "fatal"]
    report = {
        "passed": not fatal,
        "violations": violations,
        "fatal_count": len(fatal),
        "warning_count": len(violations) - len(fatal),
    }
    return {"critic_report": report}
