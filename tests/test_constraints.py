import pytest

from core.rules import rule_extract, validate_plan


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "带70岁老人和小孩去深圳玩3天，需要轮椅，饮食清淡，避开海鲜排档",
            {"with_elderly": True, "with_kids": True, "wheelchair": True, "days": 3},
        ),
        ("深圳2日游，遇到台风", {"days": 2, "avoid_outdoor_on_rain": True}),
        ("一家人去深圳玩4天，小孩3岁，人均餐标100以内", {"days": 4, "with_kids": True}),
    ],
)
def test_rule_extract(query, expected):
    got = rule_extract(query)
    for key, value in expected.items():
        assert got.get(key) == value, f"{key}: {got.get(key)} != {value}"


def test_validate_rest_slot_is_ignored():
    plan = {"slots": [{"day": 1, "period": "下午", "kind": "rest", "ref_id": "", "name": "休息"}]}
    assert validate_plan(plan, {}, {}, "") == []


def test_validate_unknown_ref_is_fatal():
    plan = {"slots": [{"day": 1, "period": "上午", "kind": "poi", "ref_id": "nope", "name": "x"}]}
    codes = [v["code"] for v in validate_plan(plan, {}, {}, "")]
    assert "UNKNOWN_REF" in codes


def test_validate_rain_outdoor():
    plan = {"slots": [{"day": 1, "period": "上午", "kind": "poi", "ref_id": "p01", "name": "深圳湾公园"}]}
    cons = {"with_elderly": True, "avoid_outdoor_on_rain": True}
    weather = {"1": {"rain": True, "level": "暴雨"}}
    codes = [v["code"] for v in validate_plan(plan, cons, weather, "")]
    assert "OUTDOOR_IN_RAIN" in codes


def test_validate_wheelchair():
    plan = {"slots": [{"day": 1, "period": "上午", "kind": "poi", "ref_id": "p15", "name": "东门老街"}]}
    codes = [v["code"] for v in validate_plan(plan, {"wheelchair": True}, {}, "")]
    assert "NO_WHEELCHAIR" in codes
