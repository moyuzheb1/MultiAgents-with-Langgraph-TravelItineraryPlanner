"""Assembler：把 Worker 召回的候选物化成具体行程槽位。

这一步是**确定性的**（不使用 LLM），只从真实召回的实体里做筛选与排布，
因此行程中不可能出现知识库外的幻觉点位。
"""
from core.rules import ENTITY_INDEX, distance_to_hotel, estimate_walk_km

PERIOD_ORDER = {"上午": 1, "午餐": 2, "下午": 3, "晚餐": 4}


def _is_poi(row: dict) -> bool:
    return row.get("label", "POI") == "POI" and row.get("id", "").startswith("p")


def _is_restaurant(row: dict) -> bool:
    return row.get("label") == "Restaurant" or row.get("id", "").startswith("r")


def _poi_score(row: dict, constraints: dict) -> tuple:
    """排序键：优先老人友好度，其次儿童友好度，再其次强度低。"""
    ent = ENTITY_INDEX.get(row.get("id", ""))
    if ent is None:
        return (0, 0, -99)
    key_elder = getattr(ent, "elder", 0) if constraints.get("with_elderly") else 0
    key_kid = getattr(ent, "kid", 0) if constraints.get("with_kids") else 0
    return (key_elder, key_kid, -getattr(ent, "intensity", 5))


def _restaurant_score(row: dict, constraints: dict) -> tuple:
    ent = ENTITY_INDEX.get(row.get("id", ""))
    if ent is None:
        return (0, 0, 0)
    light = getattr(ent, "light", 3)
    key_light = light if "清淡" in (constraints.get("dietary") or []) else 0
    key_hygiene = -getattr(ent, "hygiene_risk", 3)
    key_elder = getattr(ent, "elder", 0)
    return (key_light, key_hygiene, key_elder)


def _pick_poi(pool, constraints, area, exclude, walk_used, walk_budget, require_indoor=False):
    max_intensity = int(constraints.get("max_intensity", 5) or 5)
    cands = []
    for row in pool:
        if not _is_poi(row) or row.get("id") in exclude:
            continue
        if constraints.get("wheelchair") and not row.get("wheelchair"):
            continue
        if require_indoor and not row.get("indoor"):
            continue
        if row.get("intensity", 5) > max_intensity:
            continue
        ent = ENTITY_INDEX.get(row.get("id"))
        if ent is None:
            continue
        if walk_used + estimate_walk_km(ent) > walk_budget + 1e-6:
            continue
        in_area = (not area) or row.get("area") == area
        cands.append((1 if in_area else 0, _poi_score(row, constraints), row))

    if not cands:
        # 同区域无可用候选时放宽区域限制再试一次
        for row in pool:
            if not _is_poi(row) or row.get("id") in exclude:
                continue
            if constraints.get("wheelchair") and not row.get("wheelchair"):
                continue
            if require_indoor and not row.get("indoor"):
                continue
            if row.get("intensity", 5) > max_intensity:
                continue
            ent = ENTITY_INDEX.get(row.get("id"))
            if ent is None:
                continue
            if walk_used + estimate_walk_km(ent) > walk_budget + 1e-6:
                continue
            cands.append((0, _poi_score(row, constraints), row))

    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return cands[0][2]


def _pick_restaurant(pool, constraints, exclude, hotel_name, near_hotel=False, period="午餐"):
    cands = []
    limit_km = constraints.get("near_hotel_km")
    budget = constraints.get("budget_per_meal")
    for row in pool:
        if not _is_restaurant(row) or row.get("id") in exclude:
            continue
        if constraints.get("wheelchair") and not row.get("wheelchair"):
            continue
        if "清淡" in (constraints.get("dietary") or []) and row.get("light", 0) < 4:
            continue
        if "素食" in (constraints.get("dietary") or []) and "素食" not in (row.get("tags") or []):
            continue
        if "不吃海鲜" in (constraints.get("dietary") or []) and "海鲜" in (row.get("cuisine") or ""):
            continue
        if (constraints.get("with_elderly") or constraints.get("with_kids")) \
                and row.get("hygiene_risk", 0) >= 4:
            continue
        for avoid in (constraints.get("must_avoid") or []):
            if avoid in (row.get("tags") or []) or avoid in (row.get("cuisine") or ""):
                break
        else:
            if budget and row.get("price", 0) > budget:
                continue
            ent = ENTITY_INDEX.get(row.get("id"))
            if ent is None:
                continue
            if near_hotel and hotel_name and limit_km:
                if distance_to_hotel(ent, hotel_name) > limit_km:
                    continue
            cands.append((_restaurant_score(row, constraints), row))

    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]


def assemble(skeleton: list, constraints: dict, poi_pool: list, dining_pool: list,
             hotel_name: str = "", weather: dict = None) -> list:
    """按骨架与硬约束把候选物化成槽位列表。"""
    slots = []
    used = set(constraints.get("_exclude") or [])
    walk_limit = float(constraints.get("daily_walk_km_limit", 6.0) or 6.0)
    weather = weather or {}

    for sk in skeleton:
        day = sk.get("day", 1)
        area = sk.get("area", "")
        budget_km = float(sk.get("walk_budget_km") or walk_limit)
        walk_used = 0.0
        day_weather = weather.get(str(day), {})
        require_indoor = bool(day_weather.get("rain") and constraints.get("avoid_outdoor_on_rain"))
        max_poi = int(constraints.get("max_attractions_per_day", 3) or 3)
        poi_count = 0

        # 按 max_attractions_per_day 自主填满时段，不盲信 LLM 骨架的 periods 字段
        for period in ("上午", "下午", "晚上"):
            if poi_count >= max_poi:
                break
            row = _pick_poi(poi_pool, constraints, area, used, walk_used, budget_km,
                            require_indoor=require_indoor)
            if row is None:
                # 雨天室内候选不足时插入休息槽位，避免空日
                if require_indoor:
                    slots.append({
                        "day": day, "period": period, "kind": "rest",
                        "ref_id": "", "name": "酒店/室内休息",
                        "note": "雨天室内备选不足，建议酒店内休整或就近商场",
                    })
                break
            ent = ENTITY_INDEX[row["id"]]
            walk_used += estimate_walk_km(ent)
            used.add(row["id"])
            poi_count += 1
            slots.append({
                "day": day, "period": period, "kind": "poi",
                "ref_id": row["id"], "name": row["name"],
                "note": f"{row.get('area', '')}·强度{row.get('intensity')}·"
                        f"步行约{estimate_walk_km(ent)}km",
            })

        lunch = _pick_restaurant(dining_pool, constraints, used, hotel_name,
                                 near_hotel=False, period="午餐")
        if lunch:
            used.add(lunch["id"])
            slots.append({
                "day": day, "period": "午餐", "kind": "dining",
                "ref_id": lunch["id"], "name": lunch["name"],
                "note": f"{lunch.get('cuisine', '')}·人均{lunch.get('price')}元·"
                        f"清淡{lunch.get('light')}/5",
            })

        dinner = _pick_restaurant(dining_pool, constraints, used, hotel_name,
                                  near_hotel=True, period="晚餐")
        if not dinner:  # 就近无候选时回退到「就近优先」，交由校验器给出距离提示
            dinner = _pick_restaurant(dining_pool, constraints, used, hotel_name,
                                      near_hotel=False, period="晚餐")
        if dinner:
            used.add(dinner["id"])
            if hotel_name:
                km = distance_to_hotel(ENTITY_INDEX[dinner["id"]], hotel_name)
                near_tag = f"距酒店{km}km"
            else:
                near_tag = "近酒店"
            slots.append({
                "day": day, "period": "晚餐", "kind": "dining",
                "ref_id": dinner["id"], "name": dinner["name"],
                "note": f"{dinner.get('cuisine', '')}·人均{dinner.get('price')}元·{near_tag}",
            })

    slots.sort(key=lambda s: (s["day"], PERIOD_ORDER.get(s["period"], 9)))
    return slots


def assembler_node(state: dict) -> dict:
    """Assembler 节点：物化行程槽位（确定性）。"""
    constraints = state.get("constraints", {}) or {}
    plan = state.get("plan", {}) or {}
    workers = state.get("worker_results", {}) or {}

    poi_pool = (workers.get("poi") or {}).get("data") or []
    dining_pool = (workers.get("dining") or {}).get("data") or []
    weather = (workers.get("weather") or {}).get("data") or {}
    hotel_name = constraints.get("hotel_name", "")

    slots = assemble(plan.get("skeleton", []), constraints, poi_pool, dining_pool,
                     hotel_name, weather)

    new_plan = {**plan, "slots": slots}
    return {"plan": new_plan}
