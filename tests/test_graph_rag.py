from retrieval import area_summary, graph_rag_search

HOTEL = "深圳福田香格里拉大酒店"
SKELETON = {"skeleton": [{"area": "南山区"}, {"area": "福田区"}]}


def test_multi_hop_returns_paths():
    cons = {"with_elderly": True, "hotel_name": HOTEL, "near_hotel_km": 5, "max_intensity": 3}
    rows = graph_rag_search("南山 适合老人", "POI", 10, cons, SKELETON)
    assert rows
    assert all(r["graph_path"] for r in rows)
    assert any(r["graph_hops"] >= 1 for r in rows)
    assert any(r["graph_hops"] >= 2 for r in rows)
    assert all(r["intensity"] <= 3 for r in rows)


def test_wheelchair_constraint_enforced():
    cons = {"wheelchair": True, "hotel_name": HOTEL, "near_hotel_km": 8, "max_intensity": 5}
    rows = graph_rag_search("无障碍", "POI", 20, cons, SKELETON)
    assert rows
    assert all(r["wheelchair"] for r in rows)


def test_restaurant_must_avoid():
    cons = {
        "with_elderly": True,
        "must_avoid": ["海鲜排档"],
        "hotel_name": HOTEL,
        "near_hotel_km": 10,
    }
    rows = graph_rag_search("清淡 餐厅", "Restaurant", 20, cons, SKELETON)
    assert rows
    for r in rows:
        assert "海鲜排档" not in (r.get("tags") or [])
        assert "海鲜排档" not in (r.get("cuisine") or "")
        assert r.get("hygiene_risk", 0) < 4


def test_area_summary():
    s = area_summary("南山区")
    assert s["poi_count"] > 0
    assert s["restaurant_count"] > 0
    assert s["avg_intensity"] > 0
