"""回归评测集。

验证 about.txt 中强调的三大防御机制：
  1) Text2Cypher 防御性纠错（无效 Cypher 必被 AST/Schema 拦截）
  2) Critic 硬约束校验（老人体力 / 天气 / 无障碍 / 卫生等零容错）
  3) Critic 校验失败时的自动回跳重规划（Replan 收敛）

全部在 FAST_MODE（确定性，无需 LLM）下运行，保证可复现、可秒级回归。
运行：python evals/run.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["FAST_MODE"] = "1"
os.environ["ENABLE_LLM_CRITIC"] = "0"

from agents.assembler import assembler_node  # noqa: E402
from agents.critic import critic_node  # noqa: E402
from agents.planner import planner_node  # noqa: E402
from agents.replanner import replanner_node  # noqa: E402
from agents.router import router_node  # noqa: E402
from agents.workers import workers_node  # noqa: E402
from core.rules import rule_extract, validate_plan  # noqa: E402
from core.state import initial_state  # noqa: E402
from retrieval.graph_store import CypherError, parse_cypher  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


# ---------------- 1. 硬约束抽取 ----------------
def test_constraint_extraction() -> None:
    print("\n[1] 硬约束抽取")
    cases = [
        ("带70岁老人和小孩去深圳玩3天，需要轮椅，饮食清淡，避开海鲜排档", {
            "with_elderly": True, "with_kids": True, "wheelchair": True,
            "days": 3, "dietary": ["清淡"], "must_avoid": ["海鲜排档"]}),
        ("深圳2日游，遇到台风", {
            "days": 2, "avoid_outdoor_on_rain": True, "with_elderly": False}),
        ("一家人去深圳玩4天，小孩3岁，人均餐标100以内", {
            "days": 4, "with_kids": True, "budget_per_meal": 100}),
    ]
    for q, exp in cases:
        c = rule_extract(q)
        for k, v in exp.items():
            check(f"抽取 {k}={v} ← {q[:18]}…", c.get(k) == v,
                  f"实际 {k}={c.get(k)}")


# ---------------- 2. Text2Cypher 防御链 ----------------
def test_text2cypher_defense() -> None:
    print("\n[2] Text2Cypher 防御性校验（无效 Cypher 必被拦截）")
    bad = [
        ("臆造属性", "MATCH (n:POI) WHERE n.has_elevator = true RETURN n LIMIT 5"),
        ("未知标签", "MATCH (n:Scenic) RETURN n LIMIT 5"),
        ("未知关系", "MATCH (n:POI)-[:FRIEND_OF]->(:Area) RETURN n LIMIT 5"),
        ("写操作", "CREATE (n:POI {name:'x'}) RETURN n"),
        ("多段 MATCH", "MATCH (n:POI) MATCH (m:Restaurant) RETURN n LIMIT 5"),
    ]
    for name, cypher in bad:
        try:
            parse_cypher(cypher)
            check(f"拦截 {name}", False, "未被拦截")
        except CypherError:
            check(f"拦截 {name}", True)
    # 合法 Cypher 应通过
    ok = ("MATCH (n:POI)-[:LOCATED_IN]->(:Area {name:'南山区'}) "
          "WHERE n.wheelchair = true RETURN n LIMIT 5")
    try:
        parse_cypher(ok)
        check("放行 合法 Cypher", True)
    except CypherError as e:
        check("放行 合法 Cypher", False, str(e))


# ---------------- 3. Critic 硬约束校验 ----------------
def test_critic_rules() -> None:
    print("\n[3] Critic 硬约束校验")
    # 雨天室外
    cons = {"with_elderly": True, "avoid_outdoor_on_rain": True}
    weather = {"1": {"rain": True, "level": "暴雨"}}
    plan = {"slots": [{"day": 1, "period": "上午", "kind": "poi", "ref_id": "p01", "name": "深圳湾公园"}]}
    vs = validate_plan(plan, cons, weather, "")
    codes = [v["code"] for v in vs]
    check("雨天室外 → OUTDOOR_IN_RAIN", "OUTDOOR_IN_RAIN" in codes, str(codes))

    # 无障碍
    cons2 = {"wheelchair": True}
    plan2 = {"slots": [{"day": 1, "period": "上午", "kind": "poi", "ref_id": "p15", "name": "东门老街"}]}
    vs2 = validate_plan(plan2, cons2, {}, "")
    codes2 = [v["code"] for v in vs2]
    check("非无障碍 → NO_WHEELCHAIR", "NO_WHEELCHAIR" in codes2, str(codes2))

    # 卫生风险（海鲜排档 / 老人）
    cons3 = {"with_elderly": True}
    plan3 = {"slots": [{"day": 1, "period": "晚餐", "kind": "dining", "ref_id": "r12", "name": "蛇口海鲜大排档"}]}
    vs3 = validate_plan(plan3, cons3, {}, "")
    codes3 = [v["code"] for v in vs3]
    check("海鲜排档 → HYGIENE_RISK", "HYGIENE_RISK" in codes3, str(codes3))


# ---------------- 4. Replan 回跳收敛 ----------------
def test_replan_convergence() -> None:
    print("\n[4] Critic 校验失败 → Replan 自动回跳收敛")
    q = ("带70岁老人和小孩去深圳玩3天，需要轮椅无障碍，饮食清淡，"
         "住深圳福田香格里拉大酒店，避开海鲜排档")
    state = initial_state(q, "eval")
    state.update(router_node(state))
    state.update(planner_node(state))
    state.update(workers_node(state))
    state.update(assembler_node(state))
    slots = list(state["plan"]["slots"])
    slots = [{"day": 1, "period": "上午", "kind": "poi", "ref_id": "p04", "name": "欢乐谷"}] + slots
    slots.append({"day": 1, "period": "晚餐", "kind": "dining", "ref_id": "r12", "name": "蛇口海鲜大排档"})
    state["plan"]["slots"] = slots

    r1 = critic_node(state)["critic_report"]
    check("注入后存在致命违规", r1["fatal_count"] > 0, str(r1["fatal_count"]))
    state.update({"critic_report": r1})
    state.update(replanner_node(state))
    check("Replan 决策回跳 assembler", state["_next"] == "assembler")
    check("违规实体进入排除集", "p04" in state["constraints"].get("_exclude", [])
          and "r12" in state["constraints"].get("_exclude", []))

    state.update(assembler_node(state))
    r2 = critic_node(state)["critic_report"]
    check("第1轮回跳后致命违规清零", r2["fatal_count"] == 0, str(r2["fatal_count"]))


# ---------------- 5. 端到端 FAST_MODE ----------------
def test_e2e_fast() -> None:
    print("\n[5] 端到端（FAST_MODE，无 LLM）")
    from agents import run_trip  # noqa: E402
    queries = [
        "带70岁老人和6岁小孩去深圳玩3天，需要轮椅无障碍，饮食清淡，住深圳福田香格里拉大酒店，避开海鲜排档",
        "深圳2日游，遇到台风，带老人，要避开海鲜排档",
        "一家人去深圳玩4天，小孩3岁，人均餐标100以内",
    ]
    for q in queries:
        state = run_trip(q, session_id="eval")
        report = state.get("critic_report", {}) or {}
        slots = state.get("plan", {}).get("slots", [])
        check(f"致命违规=0 ← {q[:16]}…", report.get("fatal_count", 1) == 0,
              f"fatal={report.get('fatal_count')}")
        check(f"产出行程槽位>0 ← {q[:16]}…", len(slots) > 0, f"slots={len(slots)}")


if __name__ == "__main__":
    test_constraint_extraction()
    test_text2cypher_defense()
    test_critic_rules()
    test_replan_convergence()
    test_e2e_fast()
    print(f"\n==== 评测结果：通过 {PASS} / 失败 {FAIL} ====")
    sys.exit(1 if FAIL else 0)
