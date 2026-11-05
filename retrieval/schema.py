"""知识图谱 Schema 定义。

Text2Cypher 的「AST Schema 校验」步骤以此为准：
模型生成的标签 / 关系 / 属性只要不在白名单内，一律判为幻觉并触发重写或降级。
"""

NODE_LABELS = {
    "POI": {
        "id", "name", "category", "area", "lat", "lng", "indoor", "wheelchair",
        "intensity", "hours", "ticket", "open_hours", "kid", "elder", "tags", "desc",
    },
    "Restaurant": {
        "id", "name", "area", "lat", "lng", "cuisine", "light", "open_hours",
        "wheelchair", "price", "kid", "elder", "hygiene_risk", "tags",
    },
    "Hotel": {"id", "name", "area", "lat", "lng"},
    "Area": {"name"},
}

# 关系名 -> (允许的起始标签集合, 允许的目标标签集合)
RELATIONSHIPS = {
    "LOCATED_IN": ({"POI", "Restaurant", "Hotel"}, {"Area"}),
    "NEAR": ({"POI", "Restaurant"}, {"Hotel"}),
}

# 允许出现在 WHERE 中的比较运算符
OPERATORS = ("<=", ">=", "=", "<", ">")

# 参与向量检索的文本字段
VECTOR_TEXT_FIELDS = {
    "POI": ("name", "category", "area", "tags", "desc"),
    "Restaurant": ("name", "cuisine", "area", "tags"),
    "Hotel": ("name", "area"),
}


def label_of(entity) -> str:
    return type(entity).__name__


def schema_prompt() -> str:
    """渲染给 LLM 的 Schema 描述（用于 Text2Cypher 生成）。"""
    lines = ["知识图谱 Schema：", "节点标签及属性："]
    for label, props in NODE_LABELS.items():
        lines.append(f"  (:{label} {{ {', '.join(sorted(props))} }})")
    lines.append("关系类型：")
    lines.append("  (:POI|:Restaurant|:Hotel)-[:LOCATED_IN]->(:Area {name})")
    lines.append("  (:Restaurant|:POI)-[:NEAR {max_km: <公里>}]->(:Hotel {name})")
    lines.append("")
    lines.append("只支持以下查询形式（超出范围会被判为语法错误）：")
    lines.append("  MATCH (n:POI)-[:LOCATED_IN]->(:Area {name:'南山区'})")
    lines.append("  WHERE n.wheelchair = true AND n.intensity <= 3")
    lines.append("  [ORDER BY n.elder DESC] [LIMIT 10]")
    lines.append("约束：仅支持单段 MATCH；比较运算符限 = / <= / >= / < / > / CONTAINS；"
                 "不要使用 IN 列表或多段 MATCH；字符串用 = 或 CONTAINS（如 n.cuisine CONTAINS '粤菜'）。")
    try:
        from data.shenzhen import HOTELS
        hotel_names = " / ".join(h.name for h in HOTELS)
    except Exception:
        hotel_names = "深圳福田香格里拉大酒店"
    lines.append(f"可用酒店名：{hotel_names}")
    return "\n".join(lines)
