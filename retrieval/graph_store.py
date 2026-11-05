"""图存储层：Neo4j 优先，无 Neo4j 时回退到进程内图。

进程内图实现了一个**受控的 Cypher 子集解析器**：
凡是不在 Schema 白名单内的标签 / 关系 / 属性，或超出子集范围的语法，都会抛出
CypherError，从而被 Text2Cypher 的 Fallback Pipeline 捕获并触发重写或降级。
"""
import dataclasses
import os
import re
from typing import Any, Optional

from core.geo import haversine_km
from data.shenzhen import HOTELS, POIS, RESTAURANTS
from retrieval.schema import NODE_LABELS, RELATIONSHIPS, label_of

# 明确拒绝的写操作与高级语法
FORBIDDEN = ("CREATE", "MERGE", "DELETE", "DETACH", "SET ", "REMOVE", "CALL ", "UNWIND", "WITH ")


class CypherError(Exception):
    """Cypher 语法错误或 Schema 校验失败。"""


class ParsedQuery:
    __slots__ = ("label", "area", "conditions", "near_hotel", "near_max_km", "order_by", "limit")

    def __init__(self):
        self.label = ""
        self.area: Optional[str] = None
        self.conditions: list = []
        self.near_hotel: Optional[str] = None
        self.near_max_km: Optional[float] = None
        self.order_by: Optional[tuple] = None
        self.limit: int = 20


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        raise CypherError(f"无法解析的取值：{raw}")


def parse_cypher(cypher: str) -> ParsedQuery:
    """解析并校验 Cypher 子集，失败时抛出 CypherError。"""
    text = " ".join(cypher.strip().split())
    upper = text.upper()

    for kw in FORBIDDEN:
        if kw in upper:
            raise CypherError(f"不支持的 Cypher 语法：{kw.strip()}")

    if text.count("MATCH") > 1:
        raise CypherError("不支持多段 MATCH 模式")

    m = re.match(r"MATCH\s*\((\w+)\s*:\s*(\w+)\s*\)(.*?)(?=\bWHERE\b|\bRETURN\b|\bORDER\b|\bLIMIT\b|$)",
                 text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise CypherError("无法解析 MATCH 子句，期望形如 MATCH (n:POI)")
    var, label, chain = m.group(1), m.group(2), m.group(3)

    label = next((k for k in NODE_LABELS if k.lower() == label.lower()), label)
    if label not in NODE_LABELS:
        raise CypherError(f"未知节点标签 {label}，可用：{list(NODE_LABELS)}")

    q = ParsedQuery()
    q.label = label

    # 关系链：-[:LOCATED_IN]->(:Area {name:'南山区'})
    for rel in re.finditer(
        r"-\[\s*:\s*(\w+)\s*(?:\{([^}]*)\})?\s*\]\s*->\s*\(\s*\w*\s*:\s*(\w+)\s*(?:\{([^}]*)\})?\s*\)",
        chain,
    ):
        rel_name, rel_props, tgt_label, tgt_props = rel.groups()
        rel_name = rel_name.upper()
        if rel_name not in RELATIONSHIPS:
            raise CypherError(f"未知关系类型 {rel_name}，可用：{list(RELATIONSHIPS)}")
        src_ok, tgt_ok = RELATIONSHIPS[rel_name]
        if label not in src_ok:
            raise CypherError(f"(:{label}) 不能作为 [: {rel_name}] 的起始节点")
        if tgt_label not in tgt_ok:
            raise CypherError(f"[: {rel_name}] 不能指向 (:{tgt_label})")

        if rel_name == "LOCATED_IN":
            nm = re.search(r"name\s*:\s*'([^']+)'", tgt_props or "")
            if nm:
                q.area = nm.group(1)
        elif rel_name == "NEAR":
            nm = re.search(r"name\s*:\s*'([^']+)'", tgt_props or "")
            if nm:
                q.near_hotel = nm.group(1)
            km = re.search(r"max_km\s*:\s*([\d.]+)", rel_props or "")
            if km:
                q.near_max_km = float(km.group(1))

    # WHERE 条件
    mw = re.search(r"\bWHERE\s+(.*?)(?=\bRETURN\b|\bORDER\b|\bLIMIT\b|$)", text, re.IGNORECASE)
    if mw:
        for part in re.split(r"\s+AND\s+", mw.group(1).strip(), flags=re.IGNORECASE):
            cm = re.match(rf"{re.escape(var)}\.(\w+)\s*(<=|>=|=|<|>|CONTAINS)\s*(.+?)\s*$", part.strip())
            if not cm:
                raise CypherError(f"无法解析的 WHERE 条件：{part.strip()}")
            prop, op, raw_val = cm.group(1), cm.group(2), cm.group(3)
            if prop not in NODE_LABELS[label]:
                raise CypherError(f"(:{label}) 不存在属性 {prop}")
            q.conditions.append((prop, op, _parse_value(raw_val)))

    mo = re.search(r"\bORDER\s+BY\s+\w+\.(\w+)\s*(ASC|DESC)?", text, re.IGNORECASE)
    if mo:
        prop = mo.group(1)
        if prop not in NODE_LABELS[label]:
            raise CypherError(f"(:{label}) 不存在属性 {prop}")
        q.order_by = (prop, (mo.group(2) or "ASC").upper())

    ml = re.search(r"\bLIMIT\s+(\d+)", text, re.IGNORECASE)
    if ml:
        q.limit = int(ml.group(1))

    return q


def _compare(value: Any, op: str, target: Any) -> bool:
    try:
        if op == "=":
            if isinstance(target, bool) or isinstance(value, bool):
                return bool(value) == bool(target)
            return value == target
        if op == "CONTAINS":
            return str(target) in str(value)
        if op == "<=":
            return value <= target
        if op == ">=":
            return value >= target
        if op == "<":
            return value < target
        return value > target
    except TypeError:
        raise CypherError(f"类型不匹配，无法比较 {value!r} {op} {target!r}")


class InMemoryGraphStore:
    """进程内图：按 Schema 预建 NEAR 边（3km 内），并执行受控 Cypher 子集。"""

    def __init__(self, entities=None):
        if entities is None:
            entities = list(POIS) + list(RESTAURANTS) + list(HOTELS)
        self.entities = entities
        self.hotels = {h.name: h for h in HOTELS}

    def _candidates(self, label: str) -> list:
        return [e for e in self.entities if label_of(e) == label]

    def execute(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        q = parse_cypher(cypher)
        rows = self._candidates(q.label)

        if q.area:
            rows = [e for e in rows if getattr(e, "area", None) == q.area]

        if q.near_hotel:
            hotel = self.hotels.get(q.near_hotel)
            if hotel is None:
                raise CypherError(f"未知酒店名：{q.near_hotel}")
            max_km = q.near_max_km if q.near_max_km is not None else 3.0
            rows = [
                e for e in rows
                if haversine_km(e.lat, e.lng, hotel.lat, hotel.lng) <= max_km
            ]

        for prop, op, target in q.conditions:
            rows = [e for e in rows if _compare(getattr(e, prop, None), op, target)]

        if q.order_by:
            prop, direction = q.order_by
            rows = sorted(rows, key=lambda e: getattr(e, prop, 0),
                          reverse=(direction == "DESC"))

        rows = rows[: q.limit]
        return [{"label": label_of(e), **dataclasses.asdict(e)} for e in rows]

    @property
    def backend(self) -> str:
        return "memory"


class Neo4jGraphStore:
    """Neo4j 图存储。需要 neo4j 包与 NEO4J_URI 环境变量。"""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # 延迟导入

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()

    def execute(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def close(self) -> None:
        self._driver.close()

    @property
    def backend(self) -> str:
        return "neo4j"


def get_graph_store():
    """按环境选择图后端：配置了 NEO4J_URI 且连通则用 Neo4j，否则用进程内图。"""
    uri = os.getenv("NEO4J_URI", "").strip()
    if uri:
        try:
            return Neo4jGraphStore(
                uri,
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", ""),
            )
        except Exception:
            pass
    return InMemoryGraphStore()
