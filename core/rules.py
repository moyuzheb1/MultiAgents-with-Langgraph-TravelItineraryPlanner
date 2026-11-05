"""硬约束抽取与校验规则。

这里承载 Critic 的**确定性校验**部分：与 LLM 语义复核互为补充，
保证“老人行动力阈值 / 餐厅营业状态 / 天气预警 / 无障碍 / 卫生风险”等硬指标零容错。
"""
import re

from data.shenzhen import HOTELS, POIS, RESTAURANTS

ENTITY_INDEX = {e.id: e for e in list(POIS) + list(RESTAURANTS) + list(HOTELS)}

AREAS = ["南山区", "福田区", "罗湖区", "盐田区", "龙岗区", "宝安区", "龙华区", "光明区", "大鹏新区"]

# 各时段的代表时刻（用于营业时间校验）
PERIOD_HOUR = {"上午": 10, "午餐": 12, "下午": 15, "晚餐": 18}

# 老人/儿童同行的默认体力预算
DEFAULT_WALK_KM = {"elderly": 3.0, "kid": 4.0, "none": 6.0}
DEFAULT_MAX_INTENSITY = {"elderly": 3, "kid": 4, "none": 5}
DEFAULT_MAX_POI_PER_DAY = {"elderly": 2, "kid": 2, "none": 3}


# ---------------- 约束抽取（规则版，用于 FAST_MODE 与 LLM 兜底） ----------------

def extract_hotel(query: str) -> str:
    """识别用户提到的酒店，未提及时使用默认酒店（福田香格里拉）。"""
    text = query or ""
    for h in HOTELS:
        if h.name in text:
            return h.name
    for kw in ("福田", "会展中心"):
        if kw in text:
            return HOTELS[0].name
    if "大梅沙" in text or "盐田" in text:
        return HOTELS[2].name
    if "深圳湾" in text or "南山" in text:
        return HOTELS[1].name
    return HOTELS[0].name


def rule_extract(query: str) -> dict:
    """从自然语言中用规则抽取硬约束。"""
    c: dict = {}
    text = query or ""
    hotel = extract_hotel(text)
    defaulted = not any(h.name in text for h in HOTELS) and not any(
        kw in text for kw in ("福田", "大梅沙", "盐田", "深圳湾", "南山"))
    c["hotel_name"] = hotel
    c["_hotel_defaulted"] = defaulted

    with_elderly = bool(re.search(r"老人|父母|爸妈|爷爷|奶奶|外公|外婆|长辈|70岁|80岁", text))
    with_kids = bool(re.search(r"小孩|孩子|儿童|宝宝|带娃|亲子|婴儿|幼儿| toddler", text, re.I))
    wheelchair = bool(re.search(r"轮椅|无障碍|推车|婴儿车", text))

    c["with_elderly"] = with_elderly
    c["with_kids"] = with_kids
    c["wheelchair"] = wheelchair

    profile = "elderly" if with_elderly else ("kid" if with_kids else "none")
    c["daily_walk_km_limit"] = DEFAULT_WALK_KM[profile]
    c["max_intensity"] = DEFAULT_MAX_INTENSITY[profile]
    c["max_attractions_per_day"] = DEFAULT_MAX_POI_PER_DAY[profile]

    dietary = []
    if re.search(r"清淡|不吃辣|少油|少盐|养胃", text):
        dietary.append("清淡")
    if re.search(r"素食|吃素", text):
        dietary.append("素食")
    if re.search(r"不吃海鲜|海鲜过敏", text):
        dietary.append("不吃海鲜")
    c["dietary"] = dietary

    avoid = []
    if re.search(r"排档|避坑|不卫生|宰客|价格不透明", text):
        avoid.append("海鲜排档")
    if re.search(r"网红|排队", text):
        avoid.append("排队网红店")
    c["must_avoid"] = avoid

    c["avoid_outdoor_on_rain"] = bool(re.search(r"雨|台风|暴雨|雷", text)) or bool(with_elderly or with_kids)

    m = re.search(r"(\d+)\s*(?:天|日)(?:游|行程|的)?", text)
    c["days"] = int(m.group(1)) if m else 3

    km = re.search(r"(?:酒店|住处)\s*(?:附近|周边)?\s*([\d.]+)\s*(?:km|公里)", text)
    c["near_hotel_km"] = float(km.group(1)) if km else 5.0

    budget = re.search(r"人均.{0,4}?(\d+)\s*(?:元|块)?\s*(?:以内|以下)?", text)
    c["budget_per_meal"] = int(budget.group(1)) if budget else 150

    c["areas"] = [a for a in AREAS if a in text]
    return c


def merge_constraints(base: dict, patch: dict) -> dict:
    """用 patch 覆盖 base，空值不覆盖。"""
    merged = dict(base or {})
    for k, v in (patch or {}).items():
        if v in (None, "", []):
            continue
        merged[k] = v
    return merged


# ---------------- 校验工具 ----------------

def estimate_walk_km(poi) -> float:
    """按“时长 × 强度”估算单个景点的步行距离（公里）。"""
    hours = getattr(poi, "hours", 0) or 0
    intensity = getattr(poi, "intensity", 3) or 3
    return round(hours * (0.4 + 0.35 * (intensity - 1)), 2)


def is_open(open_hours: str, hour: int) -> bool:
    """判断营业时间是否覆盖给定小时，支持「全天」与跨夜（次日）。"""
    if not open_hours or "全天" in open_hours:
        return True
    m = re.match(r"(\d{1,2}):(\d{2})\s*-\s*(?:次日)?(\d{1,2}):(\d{2})", open_hours.strip())
    if not m:
        return True
    start = int(m.group(1))
    end = int(m.group(3))
    if "次日" in open_hours:
        end += 24
    return start <= hour < end


def hotel_by_name(name: str):
    for h in HOTELS:
        if h.name == name:
            return h
    return HOTELS[0]


def distance_to_hotel(entity, hotel_name: str) -> float:
    from core.geo import haversine_km
    h = hotel_by_name(hotel_name)
    return round(haversine_km(entity.lat, entity.lng, h.lat, h.lng), 2)


def validate_plan(plan: dict, constraints: dict, weather: dict, hotel_name: str = "") -> list[dict]:
    """对物化后的行程做确定性硬约束校验，返回违规列表。"""
    violations: list[dict] = []
    slots = plan.get("slots", []) or []

    for slot in slots:
        ent = ENTITY_INDEX.get(slot.get("ref_id", ""))
        if ent is None:
            violations.append({
                "code": "UNKNOWN_REF", "severity": "fatal", "day": slot.get("day"),
                "ref_id": slot.get("ref_id", ""),
                "detail": f"行程引用了知识库中不存在的实体 {slot.get('name', '')}",
                "suggestion": "从候选池中重新选择实体",
            })
            continue

        day = slot.get("day", 0)
        period = slot.get("period", "")
        label = type(ent).__name__

        # 1. 无障碍
        if constraints.get("wheelchair") and not getattr(ent, "wheelchair", False):
            violations.append({
                "code": "NO_WHEELCHAIR", "severity": "fatal", "day": day, "ref_id": ent.id,
                "detail": f"{ent.name} 不具备无障碍通行条件",
                "suggestion": "替换为 wheelchair=true 的同类场所",
            })

        if label == "POI":
            # 2. 单景点体力强度上限
            max_intensity = constraints.get("max_intensity", 5)
            if getattr(ent, "intensity", 0) > max_intensity:
                violations.append({
                    "code": "INTENSITY_OVER", "severity": "fatal", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 体力强度 {ent.intensity} 超过上限 {max_intensity}",
                    "suggestion": "替换为强度更低的景点或改为远观/车游",
                })

            # 3. 雨天室外
            day_weather = (weather or {}).get(str(day), {})
            rainy = day_weather.get("rain", False)
            if rainy and constraints.get("avoid_outdoor_on_rain") and not getattr(ent, "indoor", False):
                level = day_weather.get("level", "降雨")
                violations.append({
                    "code": "OUTDOOR_IN_RAIN", "severity": "fatal", "day": day, "ref_id": ent.id,
                    "detail": f"第 {day} 天有{level}，{ent.name} 为室外场所",
                    "suggestion": "替换为室内场所（展馆/商场/科技馆）",
                })

            # 4. 儿童友好度
            if constraints.get("with_kids") and getattr(ent, "kid", 3) <= 2:
                violations.append({
                    "code": "KID_UNFRIENDLY", "severity": "warning", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 对儿童友好度偏低（{ent.kid}/5）",
                    "suggestion": "可保留但需缩短停留时间",
                })

        if label == "Restaurant":
            # 5. 营业时间
            hour = PERIOD_HOUR.get(period, 12)
            if not is_open(getattr(ent, "open_hours", ""), hour):
                violations.append({
                    "code": "CLOSED", "severity": "fatal", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 营业时间 {ent.open_hours} 不覆盖{period}（约 {hour}:00）",
                    "suggestion": "改选同时段营业的餐厅",
                })

            # 6. 饮食要求
            if "清淡" in (constraints.get("dietary") or []) and getattr(ent, "light", 5) < 4:
                violations.append({
                    "code": "NOT_LIGHT", "severity": "warning", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 清淡指数 {ent.light}/5，不满足清淡要求",
                    "suggestion": "改选清淡指数 ≥4 的餐厅",
                })
            if "素食" in (constraints.get("dietary") or []) and "素食" not in (getattr(ent, "tags", []) or []):
                violations.append({
                    "code": "NOT_VEGETARIAN", "severity": "warning", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 非素食餐厅",
                    "suggestion": "改选素食餐厅",
                })
            if "不吃海鲜" in (constraints.get("dietary") or []) and "海鲜" in getattr(ent, "cuisine", ""):
                violations.append({
                    "code": "SEAFOOD", "severity": "fatal", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 主营海鲜，与「不吃海鲜」冲突",
                    "suggestion": "改选非海鲜餐厅",
                })

            # 7. 卫生风险（海鲜排档避坑）
            hygiene = getattr(ent, "hygiene_risk", 1)
            if hygiene >= 4 and (constraints.get("with_elderly") or constraints.get("with_kids")):
                violations.append({
                    "code": "HYGIENE_RISK", "severity": "fatal", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 卫生风险等级 {hygiene}/5，不适合老人/儿童",
                    "suggestion": "替换为 hygiene_risk ≤2 的明厨餐厅",
                })
            for avoid in (constraints.get("must_avoid") or []):
                if avoid in (getattr(ent, "tags", []) or []) or avoid in getattr(ent, "cuisine", ""):
                    violations.append({
                        "code": "AVOID_HIT", "severity": "fatal", "day": day, "ref_id": ent.id,
                        "detail": f"{ent.name} 命中用户明确规避项「{avoid}」",
                        "suggestion": "替换为非规避类餐厅",
                    })

            # 8. 餐标
            budget = constraints.get("budget_per_meal")
            if budget and getattr(ent, "price", 0) > budget:
                violations.append({
                    "code": "OVER_BUDGET", "severity": "warning", "day": day, "ref_id": ent.id,
                    "detail": f"{ent.name} 人均 {ent.price} 元，超出餐标 {budget} 元",
                    "suggestion": "改选人均更低的餐厅",
                })

            # 9. 距酒店距离
            if hotel_name:
                km = distance_to_hotel(ent, hotel_name)
                limit = constraints.get("near_hotel_km")
                if limit and km > limit:
                    violations.append({
                        "code": "FAR_FROM_HOTEL", "severity": "warning", "day": day, "ref_id": ent.id,
                        "detail": f"{ent.name} 距酒店 {km} km，超出 {limit} km",
                        "suggestion": "选择更近的餐厅，或理解为仅晚餐需就近",
                    })

    # 10. 单日步行量与景点数量
    days = {}
    for s in slots:
        ent = ENTITY_INDEX.get(s.get("ref_id", ""))
        if ent is None or type(ent).__name__ != "POI":
            continue
        d = s.get("day", 0)
        days.setdefault(d, []).append(ent)

    walk_limit = constraints.get("daily_walk_km_limit", 6.0)
    poi_limit = constraints.get("max_attractions_per_day", 3)
    for d, pois in days.items():
        total = round(sum(estimate_walk_km(p) for p in pois), 2)
        if total > walk_limit:
            violations.append({
                "code": "ELDER_STAMINA", "severity": "fatal", "day": d, "ref_id": "",
                "detail": f"第 {d} 天步行量约 {total} km，超出上限 {walk_limit} km",
                "suggestion": "减少当日景点或替换为低强度场所",
            })
        if len(pois) > poi_limit:
            violations.append({
                "code": "TOO_MANY_POI", "severity": "fatal", "day": d, "ref_id": "",
                "detail": f"第 {d} 天安排 {len(pois)} 个景点，超出上限 {poi_limit}",
                "suggestion": "将部分景点移至其他日期或删除",
            })

    # 11. 空日检查（既无景点也无休息安排）
    day_kinds: dict = {}
    for s in slots:
        day_kinds.setdefault(s.get("day", 0), set()).add(s.get("kind"))
    for d, kinds in day_kinds.items():
        if "poi" not in kinds and "rest" not in kinds:
            violations.append({
                "code": "EMPTY_DAY", "severity": "warning", "day": d, "ref_id": "",
                "detail": f"第 {d} 天未安排任何景点或休息",
                "suggestion": "补充室内备选或将该日设为自由活动",
            })

    return violations
