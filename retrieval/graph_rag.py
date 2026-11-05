"""多跳 GraphRAG 检索：图关系遍历 + 向量相似度的混合排序。

与单跳 Cypher 子集查询不同，本模块在实体图上做**两跳**推理：

    area  <-[LOCATED_IN]-  poi
    poi/restaurant  -[NEAR]->  hotel
    poi_area  -[LOCATED_IN]->  area  <-[LOCATED_IN]-  restaurant

每条召回结果都会附带 `graph_path`（经过的关系路径）与 `graph_hops`（跳数），
因此它既能作为默认（确定性）检索路径，也能作为 Text2Cypher 降级后的兜底，
并可在 UI 的「检索链路」里直接展示推理依据。

`area_summary()` 提供区域级的社区摘要（数量/室内占比/平均强度/高频标签），
对应 GraphRAG 中「community summary」的思路。
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Optional

from core.geo import haversine_km
from data.shenzhen import HOTELS, POIS, RESTAURANTS, all_entities
from retrieval.schema import label_of


def _entity_by_id() -> dict:
    return {e.id: e for e in all_entities()}


def _hotel_by_name(name: str):
    if not name:
        return None
    return next((h for h in HOTELS if h.name == name), None)


def _anchors(constraints: dict, plan: dict) -> tuple[set, Optional[object]]:
    """锚点：骨架/约束里的区域 + 用户酒店实体。"""
    areas = set(constraints.get("areas") or [])
    for sk in (plan.get("skeleton") or []):
        if sk.get("area"):
            areas.add(sk["area"])
    hotel = _hotel_by_name(constraints.get("hotel_name", ""))
    return areas, hotel


def _constraint_fit(entity, constraints: dict, label: str) -> tuple[bool, float]:
    """约束可行性 + 契合分（0~1）。不满足硬约束直接淘汰。"""
    score = 0.5

    if constraints.get("wheelchair") and not getattr(entity, "wheelchair", False):
        return False, 0.0

    if label == "POI":
        if getattr(entity, "intensity", 0) > int(constraints.get("max_intensity", 5) or 5):
            return False, 0.0
        if constraints.get("with_elderly"):
            score += 0.20 * (getattr(entity, "elder", 3) or 3) / 5.0
        if constraints.get("with_kids"):
            score += 0.20 * (getattr(entity, "kid", 3) or 3) / 5.0
        if constraints.get("avoid_outdoor_on_rain") and getattr(entity, "indoor", False):
            score += 0.10
    else:
        dietary = constraints.get("dietary") or []
        tags = getattr(entity, "tags", []) or []
        cuisine = getattr(entity, "cuisine", "") or ""
        if "清淡" in dietary and getattr(entity, "light", 0) < 4:
            return False, 0.0
        if "素食" in dietary and "素食" not in tags:
            return False, 0.0
        if "不吃海鲜" in dietary and "海鲜" in cuisine:
            return False, 0.0
        for avoid in (constraints.get("must_avoid") or []):
            if avoid in tags or avoid in cuisine:
                return False, 0.0
        if (constraints.get("with_elderly") or constraints.get("with_kids")) \
                and getattr(entity, "hygiene_risk", 0) >= 4:
            return False, 0.0
        if "清淡" in dietary:
            score += 0.15
        budget = constraints.get("budget_per_meal")
        if budget and getattr(entity, "price", 0) > budget:
            score -= 0.20

    return True, max(0.0, min(1.0, score))


def _to_row(entity, label: str, score: float, hops: int, path: list[str]) -> dict:
    row = dataclasses.asdict(entity) if dataclasses.is_dataclass(entity) else dict(entity.__dict__)
    row.update({
        "label": label,
        "graph_score": round(score, 4),
        "graph_hops": hops,
        "graph_path": list(path),
    })
    return row


def graph_rag_search(
    question: str,
    label_hint: str = "POI",
    top_k: int = 20,
    constraints: Optional[dict] = None,
    plan: Optional[dict] = None,
    vector_store=None,
    graph_store=None,
) -> list[dict]:
    """两跳图遍历 + 向量混合排序，返回带路径证据的候选实体。"""
    constraints = constraints or {}
    plan = plan or {}
    label = "Restaurant" if label_hint == "Restaurant" else "POI"
    areas, hotel = _anchors(constraints, plan)
    near_km = float(constraints.get("near_hotel_km") or 5.0)

    # 向量分（用于混合排序；失败不影响图检索）
    vmap: dict[str, float] = {}
    if vector_store is not None:
        try:
            for hit in vector_store.search(question, top_k=max(top_k * 3, 30), label=label):
                vmap[hit.entity.id] = hit.score
        except Exception:
            vmap = {}

    scored: list[tuple[float, float, object, list[str], int]] = []
    for ent in all_entities():
        if label_of(ent) != label:
            continue

        ok, fit = _constraint_fit(ent, constraints, label)
        if not ok:
            continue

        hops = 0
        path: list[str] = []
        bonus = 0.0

        # 1 跳：区域锚点  area <-[LOCATED_IN]- entity
        if areas and getattr(ent, "area", None) in areas:
            hops = max(hops, 1)
            path.append(f"(:{label} {{id:'{ent.id}'}})-[:LOCATED_IN]->(:Area {{name:'{ent.area}'}})")
            bonus += 0.50

        # 2 跳：酒店邻近  entity -[NEAR]-> hotel
        if hotel is not None:
            km = haversine_km(ent.lat, ent.lng, hotel.lat, hotel.lng)
            if km <= near_km:
                hops = max(hops, 2)
                path.append(f"(:{label} {{id:'{ent.id}'}})-[:NEAR {{km:{km:.1f}}}]->(:Hotel {{name:'{hotel.name}'}})")
                bonus += 0.60
            elif getattr(ent, "area", None) == getattr(hotel, "area", None):
                hops = max(hops, 2)
                path.append(f"(:{label} {{id:'{ent.id}'}})-[:SAME_AREA]->(:Hotel {{name:'{hotel.name}'}})")
                bonus += 0.20

        # 2 跳：餐厅通过「与候选景点同区」建立关系
        if label == "Restaurant" and areas and getattr(ent, "area", None) in areas:
            path.append(f"(:POI)-[:LOCATED_IN]->(:Area {{name:'{ent.area}'}})<-[:LOCATED_IN]-(:Restaurant {{id:'{ent.id}'}})")
            bonus += 0.30

        if not path:
            # 与任何锚点都无图关系，交给纯向量路径处理
            continue

        score = 0.45 * bonus + 0.35 * fit + 0.20 * vmap.get(ent.id, 0.0)
        scored.append((score, fit, ent, path, hops))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    rows = [_to_row(e, label, s, h, p) for s, _f, e, p, h in scored[:top_k]]

    # 兜底：图上确实没有可用候选时退回向量召回
    if not rows and vector_store is not None:
        try:
            rows = [
                {**hit.as_dict(), "graph_score": 0.0, "graph_hops": 0, "graph_path": []}
                for hit in vector_store.search(question, top_k=top_k, label=label)
            ]
        except Exception:
            rows = []
    return rows


def area_summary(area: str) -> dict:
    """区域社区摘要（GraphRAG community summary 的轻量实现）。"""
    pois = [p for p in POIS if p.area == area]
    rests = [r for r in RESTAURANTS if r.area == area]
    avg_intensity = round(sum(p.intensity for p in pois) / len(pois), 2) if pois else 0.0
    tags = Counter(t for p in pois for t in (p.tags or []))
    return {
        "area": area,
        "poi_count": len(pois),
        "restaurant_count": len(rests),
        "indoor_poi": sum(1 for p in pois if p.indoor),
        "wheelchair_poi": sum(1 for p in pois if p.wheelchair),
        "avg_intensity": avg_intensity,
        "top_tags": tags.most_common(5),
    }
