import pytest

from agents import run_trip

QUERIES = [
    "带70岁老人和6岁小孩去深圳玩3天，需要轮椅无障碍，饮食清淡，"
    "住深圳福田香格里拉大酒店，避开海鲜排档",
    "深圳2日游，遇到台风，带老人，要避开海鲜排档",
    "一家人去深圳玩4天，小孩3岁，人均餐标100以内",
]


@pytest.mark.parametrize("query", QUERIES)
def test_e2e_fast_mode(query):
    state = run_trip(query, session_id="pytest")
    report = state.get("critic_report", {})
    slots = state.get("plan", {}).get("slots", [])
    assert report.get("fatal_count", 1) == 0
    assert len(slots) > 0


def test_workers_use_graph_rag_in_fast_mode():
    state = run_trip(QUERIES[0], session_id="pytest")
    sources = {p.get("source") for p in (state.get("worker_results") or {}).values()}
    assert "graph_rag" in sources


def test_replan_history_has_unique_rounds():
    state = run_trip(QUERIES[1], session_id="pytest")
    rounds = [h.get("round") for h in state.get("replan_history", [])]
    assert len(rounds) == len(set(rounds))
    assert rounds == sorted(rounds)
