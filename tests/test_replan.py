from agents.replanner import replanner_node


def _state(replan_count: int) -> dict:
    return {
        "constraints": {},
        "critic_report": {
            "violations": [
                {"code": "ELDER_STAMINA", "severity": "fatal", "ref_id": "p01"},
                {"code": "TOO_MANY_POI", "severity": "warning", "ref_id": ""},
            ],
            "fatal_count": 1,
        },
        "replan_count": replan_count,
        "replan_history": [],
    }


def test_local_replan_excludes_entity():
    out = replanner_node(_state(0))
    assert out["_next"] == "assembler"
    assert "p01" in out["constraints"]["_exclude"]
    # 归约器负责累积，这里只返回本轮新增的一条
    assert len(out["replan_history"]) == 1
    assert out["constraints"].get("_full_replan") is None


def test_escalates_to_full_replan_on_third_round():
    out = replanner_node(_state(2))
    assert out["_next"] == "planner"
    assert out["constraints"]["_full_replan"] is True
