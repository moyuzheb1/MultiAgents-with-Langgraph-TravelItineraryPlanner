"""深圳家庭出行管家（Streamlit 界面）。

极简杂志风：报头 + 细分隔线 + 大号衬线标题，无圆角、无阴影、无 emoji；
顶部概览数据 + 卫星实景地图 + 按天编排的行程版式（含实景缩略图与门票/交通/无障碍信息），
下方分栏为推荐依据 / 校验记录 / 编排说明。
"""
from __future__ import annotations

import base64
import colorsys
import html
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import folium
import streamlit as st
from streamlit_folium import st_folium

from agents import stream_trip  # noqa: E402
from core.llm import get_api_key  # noqa: E402
from core.rules import ENTITY_INDEX  # noqa: E402
from data.shenzhen import HOTELS, POIS  # noqa: E402
from retrieval import area_summary  # noqa: E402

EXAMPLES = [
    ("老年无障碍", "带70岁老人和6岁小孩去深圳玩3天，需要轮椅无障碍，饮食清淡，"
                 "住深圳福田香格里拉大酒店，避开海鲜排档"),
    ("台风天备选", "深圳2日游，遇到台风，带老人，要避开海鲜排档"),
    ("低龄亲子", "一家人去深圳玩4天，小孩3岁，人均餐标100以内"),
    ("大梅沙休闲", "带老人去大梅沙附近玩2天，不要爬山，雨天要有室内备选"),
]

AREAS = ["南山区", "福田区", "罗湖区", "盐田区", "龙岗区", "宝安区", "龙华区", "光明区", "坪山区", "大鹏新区"]

PERIOD_ORDER = {"上午": 1, "午餐": 2, "下午": 3, "晚餐": 4, "晚上": 5}
PERIOD_TIME = {"上午": "09:00", "午餐": "12:00", "下午": "15:00", "晚餐": "18:00", "晚上": "20:00"}
KIND_META = {"poi": ("景点", "poi"), "dining": ("餐饮", "dining"), "rest": ("休息", "rest")}

PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "photos")


@st.cache_data(show_spinner=False, max_entries=1024)
def photo_data_uri(ref_id: str) -> str:
    """读取 static/photos/<实体 id>.jpg，返回 data URI；缺失时返回空串。"""
    if not ref_id:
        return ""
    for folder in (os.path.join(PHOTO_DIR, "thumbs"), PHOTO_DIR):
        try:
            with open(os.path.join(folder, f"{ref_id}.jpg"), "rb") as f:
                return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")
        except OSError:
            continue
    return ""

# 卫星影像上要足够醒目，所以取高对比度但不过分鲜艳的一组
DAY_COLORS = [
    [230, 81, 0],
    [21, 101, 192],
    [194, 24, 91],
    [0, 121, 107],
    [93, 64, 160],
]
KIND_COLORS = {"poi": [230, 81, 0], "dining": [194, 24, 91], "hotel": [84, 84, 84]}

CSS = """
<style>
:root {
  --paper:#FBFAF8; --ink:#111111; --muted:#6B6B6B; --line:#D9D6D0; --accent:#9B2C2C;
}
.stApp {background: var(--paper);}
.block-container {max-width: 1180px; padding-top: 1.4rem; padding-bottom: 3rem;}
#MainMenu, footer {visibility: hidden;}

html, body, [class*="css"] {
  font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--ink);
}

.masthead {border-bottom: 2px solid var(--ink); padding-bottom: 12px;}
.kicker {font-size: 10px; letter-spacing: .24em; text-transform: uppercase; color: var(--muted);}
.masthead h1 {
  font-family: "Songti SC", "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
  font-size: 46px; line-height: 1.04; margin: 10px 0 8px; font-weight: 700; letter-spacing: -.5px;
}

.section-label {
  font-size: 10px; letter-spacing: .2em; text-transform: uppercase;
  color: var(--muted); border-top: 1px solid var(--ink); padding-top: 10px; margin: 6px 0 12px;
}

.kpi-row {
  display: flex; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--line);
  margin: 4px 0 24px;
}
.kpi-item {flex: 1; padding: 14px 16px; border-right: 1px solid var(--line);}
.kpi-item:last-child {border-right: none;}
.kpi-label {font-size: 10px; letter-spacing: .16em; text-transform: uppercase; color: var(--muted);}
.kpi-value {
  font-family: Georgia, "Times New Roman", serif; font-size: 30px; line-height: 1.1; margin-top: 4px;
}
.kpi-value.ok {color: #2F5D50;}
.kpi-value.danger {color: var(--accent);}
.kpi-sub {font-size: 11px; color: var(--muted); margin-top: 2px;}

.map-caption {
  font-size: 10px; letter-spacing: .2em; text-transform: uppercase; color: var(--muted);
  border-top: 1px solid var(--ink); padding-top: 10px; margin: 8px 0 10px;
}
.legend {display: flex; gap: 18px; flex-wrap: wrap; align-items: center; margin: 8px 0 26px; font-size: 11px; color: var(--muted);}
.legend-item {display: inline-flex; align-items: center; gap: 6px;}
.legend-swatch {width: 14px; height: 3px; display: inline-block;}
.legend-dot {
  width: 8px; height: 8px; border-radius: 50%; display: inline-block;
  box-shadow: 0 0 0 1px rgba(255, 255, 255, .9), 0 0 0 2px rgba(0, 0, 0, .12);
}
.legend-note {margin-left: auto; font-size: 10px; letter-spacing: .06em;}
.live-list {
  display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 6px 0 8px;
  max-height: 62px; overflow-y: auto;
}
.live-stop {
  font-size: 11px; color: var(--muted); border-top: 1px solid var(--line);
  padding-top: 5px; letter-spacing: .02em;
}
.live-stop b {color: var(--ink); font-weight: 600; letter-spacing: .1em; margin-right: 5px;}
.live-stop.current {color: var(--accent); border-top-color: var(--accent);}
.live-stop.current b {color: var(--accent);}
[data-testid="stCustomComponentV1"] {border: 1px solid var(--line);}
.map-caption-meta {font-size: 11px; color: var(--muted); letter-spacing: .04em;}

.day-head {
  display: grid; grid-template-columns: 92px 1fr; gap: 24px;
  padding-top: 22px; border-top: 1px solid var(--line);
}
.day-head.first {border-top: 1px solid var(--ink);}
.day-index {font-family: Georgia, "Times New Roman", serif; font-size: 40px; line-height: 1; color: var(--accent);}
.day-kicker {font-size: 10px; letter-spacing: .2em; text-transform: uppercase; color: var(--muted);}
.day-title {
  font-family: "Songti SC", "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
  font-size: 22px; margin: 4px 0 10px;
}
.stop {
  display: grid; grid-template-columns: 46px 88px 1fr auto; gap: 12px;
  margin-left: 92px; padding: 10px 0; border-top: 1px dotted var(--line);
}
.stop.first {border-top: none;}
.btn-pad {height: 12px;}
.stop-time {font-size: 11px; letter-spacing: .06em; color: var(--muted); padding-top: 3px;}
.stop-thumb {width: 88px; height: 60px; overflow: hidden; background: #F1EFEA;}
.stop-thumb img {width: 100%; height: 100%; object-fit: cover; display: block; filter: saturate(.92) contrast(1.02);}
.stop-thumb.ph {display: flex; align-items: center; justify-content: center; border: 1px solid var(--line);}
.stop-thumb.ph span {font-size: 10px; letter-spacing: .12em; color: var(--muted);}
.stop-name {font-size: 15px; font-weight: 600;}
.stop-note {font-size: 12px; color: var(--muted); margin-top: 2px; line-height: 1.6;}
.stop-facts {font-size: 11px; color: #8A857C; margin-top: 4px; line-height: 1.6; letter-spacing: .02em;}
.stop-kind {
  font-size: 9px; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
  border: 1px solid var(--line); padding: 2px 7px; height: fit-content; white-space: nowrap;
}
.stop-kind.dining {color: var(--accent); border-color: var(--accent);}

.viol {border-top: 1px solid var(--line); padding: 12px 0;}
.viol:first-of-type {border-top: 1px solid var(--ink);}
.viol-code {font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--accent);}
.viol-code.warn {color: #8A6A2F;}
.viol-detail {font-size: 14px; margin-top: 4px;}
.viol-sug {font-size: 12px; color: var(--muted); margin-top: 3px;}

.replan {border-top: 1px solid var(--line); padding: 12px 0;}
.replan:first-of-type {border-top: 1px solid var(--ink);}
.round-chip {font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--accent);}
.replan-meta {font-size: 12px; color: var(--muted); margin-top: 3px;}

.panel {border: 1px solid var(--line); padding: 18px 20px; margin-bottom: 16px;}
.note {font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 12px;}

/* 卫星地图随行程滚动 */
div[data-testid="stColumn"]:has(.map-sticky),
div[data-testid="column"]:has(.map-sticky) {position: sticky; top: 3.2rem; align-self: flex-start;}
.map-caption-row {display: flex; align-items: baseline; justify-content: space-between; gap: 12px;}
.focus-chip {
  font-size: 11px; color: var(--accent); border: 1px solid var(--accent);
  padding: 2px 7px; display: inline-block; margin-bottom: 6px;
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
  border: 1px solid var(--line) !important; background: transparent !important;
  color: var(--muted) !important; font-size: 11px !important; letter-spacing: .06em;
  padding: 4px 8px !important; min-height: 0 !important;
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--ink) !important; color: var(--ink) !important; background: transparent !important;
}
div[data-testid="stDeckGlJsonChart"] {margin: 0;}

/* 去除 Streamlit 默认圆角 */
div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {
  border-radius: 0 !important; border: 1px solid var(--ink); background: transparent;
  color: var(--ink); font-size: 12px; letter-spacing: .08em; padding: 8px 14px;
}
div.stButton > button:hover {background: var(--ink); color: #fff; border-color: var(--ink);}
div.stFormSubmitButton > button, div.stDownloadButton > button {
  background: var(--ink); color: #fff;
}
.stTextArea textarea {border-radius: 0 !important; border: 1px solid var(--line) !important; background: #fff !important;}
[data-testid="stMetric"] {border-radius: 0 !important;}
[data-testid="stExpander"] {border-radius: 0 !important; border: 1px solid var(--line) !important;}
[data-testid="stDataFrame"] {border-radius: 0 !important;}
.stTabs [data-baseweb="tab-list"] {gap: 26px; border-bottom: 1px solid var(--line);}
.stTabs [data-baseweb="tab"] {
  background: transparent; padding: 10px 0; font-size: 12px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted);
}
.stTabs [aria-selected="true"] {color: var(--accent) !important;}

@media (max-width: 760px) {
  .masthead h1 {font-size: 34px;}
  .kpi-row {flex-wrap: wrap;}
  .kpi-item {flex: 1 1 42%; border-right: none; border-bottom: 1px solid var(--line);}
  .day-head {grid-template-columns: 1fr; gap: 8px;}
  .day-index {font-size: 28px;}
  .stop {grid-template-columns: 46px 1fr; margin-left: 0;}
  .stop-kind, .stop-thumb {display: none;}
  div[data-testid="stColumn"]:has(.map-sticky),
  div[data-testid="column"]:has(.map-sticky) {position: static;}
}
@media (max-width: 900px) {
  div[data-testid="stHorizontalBlock"]:has(.map-sticky),
  div[data-testid="stHorizontalBlock"]:has(.live-map-anchor) {flex-direction: column;}
  div[data-testid="stHorizontalBlock"]:has(.map-sticky) > div,
  div[data-testid="stHorizontalBlock"]:has(.live-map-anchor) > div {width: 100% !important;}
}
</style>
"""


def dag_html(executed: set) -> str:
    """纯 HTML 架构 DAG（无外部依赖）。"""

    def box(nid: str, title: str, sub: str) -> str:
        active = (
            "background:#111;border-color:#111;color:#fff"
            if nid in executed
            else "background:transparent;border-color:#C9C6C0;color:#9A9A9A"
        )
        return (
            f'<div style="border:1px solid;{active};'
            f'padding:8px 12px;text-align:center;min-width:96px;font-size:12px">'
            f'<b>{title}</b><br><span style="font-size:10px">{sub}</span></div>'
        )

    def arrow(label: str = "", color: str = "#6B6B6B") -> str:
        return (
            f'<div style="align-self:center;color:{color};font-size:12px;'
            f'margin:0 6px;text-align:center">→<br><span style="font-size:9px">{label}</span></div>'
        )

    chain = "".join([
        box("router", "读需求", "人群 · 天数 · 偏好"), arrow(),
        box("planner", "排框架", "每天几处"), arrow(),
        box("workers", "查地点", "景点 · 餐饮 · 天气"), arrow(),
        box("assembler", "排具体行程", "时间与路线"), arrow(),
        box("critic", "查合规", "无障碍 · 天气 · 预算"),
    ])
    tail = (
        f'{arrow("通过", "#2F5D50")}{box("finalize", "出行程", "清单与提示")}'
        f'<div style="margin-top:14px;color:#9B2C2C;font-size:11px">'
        f'↺ 发现硬性冲突 → 自动调整 → 重新编排</div>'
    )
    return (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'font-family:sans-serif">{chain}{tail}</div>'
        f'<div style="margin-top:16px">{box("replan", "调整", "换点位")} '
        f'<span style="color:#9B2C2C;font-size:11px">按需触发，不改变你的需求</span></div>'
    )


def _fill_example(text: str) -> None:
    st.session_state["query_text"] = text


def render_kpis(state: dict, elapsed: float) -> None:
    plan = state.get("plan", {}) or {}
    slots = plan.get("slots", []) or []
    report = state.get("critic_report", {}) or {}
    constraints = state.get("constraints", {}) or {}
    days = int(plan.get("days") or constraints.get("days") or 0)
    fatal = int(report.get("fatal_count", 0) or 0)
    warn = int(report.get("warning_count", 0) or 0)

    items = [
        ("行程天数", str(days), "天", ""),
        ("行程安排", str(len(slots)), "个安排", ""),
        ("冲突提醒", str(fatal), "需处理" if fatal else "无", "ok" if fatal == 0 else "danger"),
        ("温馨提示", str(warn), "参考项", ""),
        ("生成耗时", f"{elapsed:.1f}", "秒", ""),
    ]
    cells = "".join(
        f'<div class="kpi-item"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value {tone}">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div></div>'
        for label, value, sub, tone in items
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)


_GCJ_A = 6378245.0
_GCJ_EE = 0.00669342162296594323


def _gcj_dlat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _gcj_dlng(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def _to_map(lng: float, lat: float) -> tuple[float, float]:
    """WGS-84 → GCJ-02：高德影像使用 GCJ-02，不转换会让点位整体偏移数百米。"""
    if not (73.66 < lng < 135.05 and 3.86 < lat < 53.55):
        return lng, lat
    dlat = _gcj_dlat(lng - 105.0, lat - 35.0)
    dlng = _gcj_dlng(lng - 105.0, lat - 35.0)
    rad = lat / 180.0 * math.pi
    magic = 1 - _GCJ_EE * math.sin(rad) ** 2
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_GCJ_A * (1 - _GCJ_EE)) / (magic * sqrt_magic) * math.pi)
    dlng = (dlng * 180.0) / (_GCJ_A / sqrt_magic * math.cos(rad) * math.pi)
    return lng + dlng, lat + dlat


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=512)
def _osrm_route(a: tuple, b: tuple) -> list:
    """两点之间的道路折线（WGS-84）；取不到就回退成直线。"""
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{a[0]:.6f},{a[1]:.6f};{b[0]:.6f},{b[1]:.6f}"
        "?overview=simplified&geometries=geojson"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "travel-planner-demo"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read())
        coords = data["routes"][0]["geometry"]["coordinates"]
        if len(coords) >= 2:
            return [(float(x), float(y)) for x, y in coords]
    except Exception:
        pass
    return [tuple(a), tuple(b)]


def _road_segments(points_wgs: list) -> list:
    """相邻两站各取一条道路折线（转 GCJ-02），按段返回，便于逐段配色。"""
    if not points_wgs:
        return []
    if len(points_wgs) < 2:
        return [[list(_to_map(*points_wgs[0]))]]
    pairs = list(zip(points_wgs[:-1], points_wgs[1:]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        raw = list(pool.map(lambda p: _osrm_route(p[0], p[1]), pairs))
    out = []
    for (a, b), seg in zip(pairs, raw):
        pts = seg if len(seg) >= 2 else [a, b]
        out.append([list(_to_map(x, y)) for x, y in pts])
    return out


def _segment_color(rgb: list, index: int, total: int) -> list:
    """同一天内按行进顺序由浅到深递进，让每一段都能单独看清。"""
    r, g, b = (v / 255 for v in list(rgb)[:3])
    h, light, sat = colorsys.rgb_to_hls(r, g, b)
    step = 0.0 if total <= 1 else index / (total - 1)
    h = (h + 0.05 * (step - 0.5)) % 1.0            # 色相小幅漂移，段与段更容易分辨
    top = min(0.60, light + 0.20)                  # 浅色端不超过 0.6，避免荧光
    bottom = max(0.22, light - 0.12)               # 深色端保证对比
    light = top - (top - bottom) * step
    sat = min(0.95, sat * 0.85)
    return [round(c * 255) for c in colorsys.hls_to_rgb(h, light, sat)]


def _build_map(state: dict, day_filter: int | None = None):
    plan = state.get("plan", {}) or {}
    slots = plan.get("slots", []) or []
    constraints = state.get("constraints", {}) or {}
    days = int(plan.get("days") or constraints.get("days") or 3)
    hotel = next((h for h in HOTELS if h.name == constraints.get("hotel_name")), None)

    routes, stops = [], []
    lngs, lats = [], []
    hotel_xy = _to_map(hotel.lng, hotel.lat) if hotel is not None else None
    hotel_wgs = (hotel.lng, hotel.lat) if hotel is not None else None

    for d in range(1, days + 1):
        if day_filter and d != day_filter:
            continue
        day_slots = sorted(
            [s for s in slots if s.get("day") == d and s.get("kind") != "rest"],
            key=lambda s: PERIOD_ORDER.get(s.get("period"), 9),
        )
        seq = []
        for s in day_slots:
            ent = ENTITY_INDEX.get(s.get("ref_id", ""))
            if ent is not None:
                seq.append((s.get("kind", "poi"), ent))
        if not seq:
            continue

        color = DAY_COLORS[(d - 1) % len(DAY_COLORS)]
        wgs_points = []
        if hotel_wgs is not None:
            wgs_points.append(hotel_wgs)

        for i, (kind, ent) in enumerate(seq, 1):
            x, y = _to_map(ent.lng, ent.lat)
            wgs_points.append((ent.lng, ent.lat))
            lngs.append(x)
            lats.append(y)
            stops.append({
                "day": d,
                "ref_id": ent.id,
                "position": [x, y],
                "label": str(i),
                "name": ent.name,
                "kind": kind,
                "color": color,
            })

        # 只画「酒店 → 当天各点」的单向路线；同一天按段配色，顺序一目了然
        seg_paths = _road_segments(wgs_points)
        for k, seg in enumerate(seg_paths):
            routes.append({
                "day": d,
                "seq": k + 1,
                "path": seg,
                "color": _segment_color(color, k, len(seg_paths)),
            })

    if hotel_xy is not None:
        stops.append({
            "day": 0,
            "ref_id": hotel.id,
            "position": [hotel_xy[0], hotel_xy[1]],
            "label": "住",
            "name": hotel.name,
            "kind": "hotel",
            "color": KIND_COLORS["hotel"],
        })
        lngs.append(hotel_xy[0])
        lats.append(hotel_xy[1])

    return routes, stops, lngs, lats


AMAP_SAT = "https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"
AMAP_ATTR = "AutoNavi"
FOCUS_HEX = "#d62828"

MAP_INNER_CSS = """
<style>
/* 影像压暗降饱和，让路线成为主角 */
.leaflet-tile-pane {filter:saturate(.72) contrast(.96) brightness(.98);}
.map-num {width:17px;height:17px;line-height:17px;text-align:center;
  font:600 9.5px/17px -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  color:#111111;}
.map-num.light {color:#ffffff;}
.map-arrow {width:12px;height:12px;line-height:0;transform-origin:50% 50%;}
.map-focus {transform:translate(-50%,-140%);background:#d62828;color:#fff;
  padding:3px 7px;white-space:nowrap;font:600 11px/1.3 -apple-system,"PingFang SC",
  "Microsoft YaHei",sans-serif;box-shadow:0 1px 3px rgba(0,0,0,.35);}
.leaflet-container {font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  background:#e9e6e0;}
.leaflet-tooltip {font-size:11px;border-radius:0;border:1px solid #d9d6d0;
  box-shadow:none;padding:3px 6px;}
.leaflet-control-scale-line {border-color:#8a857c;color:#8a857c;border-radius:0;}
.leaflet-control-attribution {font-size:9px;background:rgba(251,250,248,.8);}
.leaflet-control-zoom a {background:#FBFAF8;color:#111;border:1px solid #D9D6D0 !important;
  border-radius:0 !important;width:26px;height:26px;line-height:24px;font-size:14px;}
.leaflet-control-zoom a:hover {background:#111;color:#fff;border-color:#111 !important;}
.leaflet-control-zoom {border:none !important;box-shadow:0 1px 4px rgba(0,0,0,.18);}
</style>
"""


def _hex(rgb) -> str:
    r, g, b = (max(0, min(255, int(v))) for v in list(rgb)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def _focus_point():
    """当前被点选的地点（由行程卡片的定位按钮写入）。"""
    ref = st.session_state.get("focus_ref")
    if not ref:
        return None
    ent = ENTITY_INDEX.get(str(ref))
    if ent is None:
        return None
    lng, lat = _to_map(ent.lng, ent.lat)
    return {"ref_id": ent.id, "name": ent.name, "position": [lng, lat]}


def _set_focus(day: int, ref_id: str) -> None:
    st.session_state["focus_ref"] = ref_id
    st.session_state["map_day_label"] = f"第 {day} 天"


def _clear_focus() -> None:
    st.session_state.pop("focus_ref", None)


def _screen_angle(p0: list, p1: list) -> float:
    """墨卡托平面上的方向角（度，顺时针，0 为向右）。"""
    dlng = math.radians(p1[0] - p0[0])
    dy = (math.log(math.tan(math.pi / 4 + math.radians(p1[1]) / 2))
          - math.log(math.tan(math.pi / 4 + math.radians(p0[1]) / 2)))
    if not dlng and not dy:
        return 0.0
    return math.degrees(math.atan2(-dy, dlng))


def _arrow_marks(path: list, count: int = 4) -> list:
    """沿路线等距取点，返回 (lat, lng, 方向角)。"""
    n = len(path)
    if n < 2:
        return []
    if n == 2:
        p0, p1 = path[0], path[1]
        return [((p0[1] + p1[1]) / 2, (p0[0] + p1[0]) / 2, _screen_angle(p0, p1))]
    marks = []
    for k in range(1, count + 1):
        i = max(1, min(n - 1, int(n * k / (count + 1))))
        p0, p1 = path[i - 1], path[i]
        marks.append((p1[1], p1[0], _screen_angle(p0, p1)))
    return marks


def _build_folium(routes: list, stops: list, focus: dict | None, lats: list, lngs: list):
    if focus is not None:
        center = [focus["position"][1], focus["position"][0]]
        m = folium.Map(location=center, zoom_start=16, tiles=None,
                       control_scale=True, prefer_canvas=True)
    else:
        lat0 = (min(lats) + max(lats)) / 2 if lats else 22.55
        lng0 = (min(lngs) + max(lngs)) / 2 if lngs else 114.06
        m = folium.Map(location=[lat0, lng0], zoom_start=12, tiles=None,
                       control_scale=True, prefer_canvas=True)
    # 滚轮留给页面滚动，避免看行程时地图把页面"吃掉"
    m.options["scrollWheelZoom"] = False

    # 只保留卫星影像本身，不叠注记层，地图更安静
    folium.TileLayer(
        tiles=AMAP_SAT, name="卫星影像", attr=AMAP_ATTR,
        min_zoom=3, max_zoom=18, max_native_zoom=18,
    ).add_to(m)

    for r in routes:
        if len(r["path"]) < 2:
            continue
        color = _hex(r["color"])
        folium.PolyLine(
            locations=[[p[1], p[0]] for p in r["path"]],
            color=color, weight=3.4, opacity=0.92,
            tooltip=f'第 {r["day"]} 天 · 第 {r.get("seq", 1)} 段',
        ).add_to(m)
        # 每段放 1–2 个箭头标出行进方向
        arrows = 2 if len(r["path"]) >= 10 else 1
        for lat, lng, angle in _arrow_marks(r["path"], count=arrows):
            folium.Marker(
                location=[lat, lng],
                icon=folium.DivIcon(
                    html=(
                        f'<div class="map-arrow" style="transform:rotate({angle:.0f}deg)">'
                        f'<svg width="12" height="12" viewBox="0 0 12 12">'
                        f'<path d="M2.4 1.8 L10 6 L2.4 10.2 Z" fill="{color}" '
                        f'stroke="#FFFFFF" stroke-width="1.1" stroke-linejoin="round"/>'
                        f"</svg></div>"
                    ),
                    icon_size=(12, 12), icon_anchor=(6, 6),
                ),
            ).add_to(m)

    for s in stops:
        lat, lng = s["position"][1], s["position"][0]
        is_hotel = s["kind"] == "hotel"
        folium.CircleMarker(
            location=[lat, lng], radius=9 if not is_hotel else 10,
            color="#ffffff", weight=2,
            fill=True, fill_color=_hex(s["color"]), fill_opacity=1,
            tooltip=f"{s['label']} · {s['name']}",
        ).add_to(m)
        folium.Marker(
            location=[lat, lng],
            icon=folium.DivIcon(
                html=f'<div class="map-num light">{s["label"]}</div>',
                icon_size=(17, 17), icon_anchor=(8.5, 8.5),
            ),
        ).add_to(m)

    if focus is not None:
        lat, lng = focus["position"][1], focus["position"][0]
        folium.Circle(
            location=[lat, lng], radius=260, color=FOCUS_HEX, weight=1,
            fill=True, fill_color=FOCUS_HEX, fill_opacity=0.18,
        ).add_to(m)
        folium.CircleMarker(
            location=[lat, lng], radius=7, color="#ffffff", weight=2,
            fill=True, fill_color=FOCUS_HEX, fill_opacity=1,
            tooltip=focus["name"],
        ).add_to(m)
        folium.Marker(
            location=[lat, lng],
            icon=folium.DivIcon(
                html=f'<div class="map-focus">{html.escape(focus["name"])}</div>',
                icon_size=(1, 1), icon_anchor=(0, 0),
            ),
        ).add_to(m)

    if focus is None and lats and lngs:
        m.fit_bounds(
            [[min(lats), min(lngs)], [max(lats), max(lngs)]],
            padding=(38, 38),
        )

    m.get_root().html.add_child(folium.Element(MAP_INNER_CSS))
    return m


def render_map_panel(state: dict) -> None:
    """右侧常驻卫星地图：按天筛选 + 点选行程卡片后自动定位。"""
    st.markdown('<div class="map-sticky"></div>', unsafe_allow_html=True)
    plan = state.get("plan", {}) or {}
    constraints = state.get("constraints", {}) or {}
    days = int(plan.get("days") or constraints.get("days") or 3)
    labels = ["全部"] + [f"第 {d} 天" for d in range(1, days + 1)]
    if st.session_state.get("map_day_label") not in labels:
        st.session_state["map_day_label"] = labels[0]

    choice = st.radio(
        "地图范围", labels, horizontal=True, key="map_day_label",
        label_visibility="collapsed",
    )
    day_filter = None if choice == labels[0] else labels.index(choice)

    focus = _focus_point()
    routes, stops, lngs, lats = _build_map(state, day_filter)
    if not stops:
        st.markdown('<div class="note">当天没有可绘制的点位。</div>', unsafe_allow_html=True)
        return

    st.markdown(
        f'<div class="map-caption-row"><div class="map-caption">卫星路线图</div>'
        f'<div class="map-caption-meta">{html.escape(choice)} · {len(stops)} 处</div></div>',
        unsafe_allow_html=True,
    )
    m = _build_folium(routes, stops, focus, lats, lngs)
    st_folium(
        m, height=540, use_container_width=True,
        returned_objects=[], key="trip_map",
    )

    legend = "".join(
        f'<span class="legend-item"><span class="legend-dot" '
        f'style="background:{_hex(DAY_COLORS[(d - 1) % len(DAY_COLORS)])}"></span>'
        f"第 {d} 天</span>"
        for d in sorted({r["day"] for r in routes})
    )
    st.markdown(
        f'<div class="legend">{legend}'
        f'<span class="legend-note">同日由浅到深为顺序 · 箭头指向前进方向</span></div>',
        unsafe_allow_html=True,
    )

    if focus is not None:
        col_chip, col_clear = st.columns([3, 1])
        col_chip.markdown(
            f'<div class="focus-chip">已定位 · {html.escape(focus["name"])}</div>',
            unsafe_allow_html=True,
        )
        col_clear.button("清除定位", key="clear_focus", on_click=_clear_focus)
    else:
        st.markdown(
            '<div class="note">点行程中任意地点的「地图」可在图上定位，'
            "切换上方的「全部 / 第 N 天」可避免多条路线挤在一起。</div>",
            unsafe_allow_html=True,
        )


def _thumb_html(ref_id: str, fallback: str) -> str:
    uri = photo_data_uri(ref_id)
    if uri:
        return f'<div class="stop-thumb"><img src="{uri}" alt=""></div>'
    return f'<div class="stop-thumb ph"><span>{html.escape(fallback[:6])}</span></div>'


def _facts(ent) -> str:
    """门票 / 开放时间 / 地铁 / 无障碍 / 人均，取知识库里的真实字段。"""
    if ent is None:
        return ""
    facts = []
    ticket = getattr(ent, "ticket", None)
    if ticket is not None:
        facts.append("免费" if ticket == 0 else f"门票 ¥{ticket}")
    hours = getattr(ent, "open_hours", "")
    if hours:
        facts.append(hours)
    metro = getattr(ent, "metro", None)
    if metro:
        facts.append(" / ".join(metro))
    price = getattr(ent, "price", None)
    if price:
        facts.append(f"人均 ¥{price}")
    wheelchair = getattr(ent, "wheelchair", None)
    if wheelchair is not None:
        facts.append("无障碍友好" if wheelchair else "无障碍受限")
    return " · ".join(str(f) for f in facts)


def _slot_html(slot: dict, first: bool = False) -> str:
    period = slot.get("period", "")
    kind = slot.get("kind", "")
    label, cls = KIND_META.get(kind, (kind or "安排", "other"))
    ref_id = str(slot.get("ref_id") or "")
    ent = ENTITY_INDEX.get(ref_id)
    caption = getattr(ent, "category", "") or getattr(ent, "cuisine", "") or label
    facts = _facts(ent)
    facts_html = f'<div class="stop-facts">{html.escape(facts)}</div>' if facts else ""
    stop_cls = "stop first" if first else "stop"
    return (
        f'<div class="{stop_cls}">'
        f'<div class="stop-time">{PERIOD_TIME.get(period, "--:--")}</div>'
        f'{_thumb_html(ref_id, caption)}'
        f'<div><div class="stop-name">{html.escape(str(slot.get("name", "")))}</div>'
        f'<div class="stop-note">{html.escape(str(slot.get("note", "")))}</div>'
        f'{facts_html}</div>'
        f'<div class="stop-kind {cls}">{label}</div></div>'
    )


def render_itinerary(state: dict) -> None:
    plan = state.get("plan", {}) or {}
    slots = plan.get("slots", []) or []
    constraints = state.get("constraints", {}) or {}
    days = int(plan.get("days") or constraints.get("days") or 3)
    skeleton = {sk.get("day"): sk for sk in (plan.get("skeleton") or [])}
    weather = ((state.get("worker_results") or {}).get("weather") or {}).get("data") or {}

    for d in range(1, days + 1):
        day_slots = sorted(
            [s for s in slots if s.get("day") == d],
            key=lambda s: PERIOD_ORDER.get(s.get("period"), 9),
        )
        sk = skeleton.get(d, {}) or {}
        area = sk.get("area", "")
        theme = sk.get("theme", "")
        title = theme or area or f"第 {d} 天"
        w = weather.get(str(d), {}) or {}
        meta = " / ".join(x for x in [area, w.get("level", ""), w.get("temp", "")] if x)

        first_cls = " first" if d == 1 else ""
        st.markdown(
            f'<div class="day-head{first_cls}">'
            f'<div class="day-index">{d:02d}</div>'
            f'<div><div class="day-kicker">Day {d}'
            f'{" · " + html.escape(meta) if meta else ""}</div>'
            f'<div class="day-title">{html.escape(title)}</div></div></div>',
            unsafe_allow_html=True,
        )

        if not day_slots:
            st.markdown(
                '<div class="stop"><div class="stop-time">--:--</div>'
                '<div class="stop-thumb ph"><span>休息</span></div>'
                '<div><div class="stop-name">自由活动 / 酒店休息</div>'
                '<div class="stop-note">当日无满足硬约束的安排</div></div>'
                '<div class="stop-kind rest">休息</div></div>',
                unsafe_allow_html=True,
            )
            continue

        for idx, s in enumerate(day_slots):
            col_card, col_btn = st.columns([30, 3], gap="small")
            with col_card:
                st.markdown(_slot_html(s, first=idx == 0), unsafe_allow_html=True)
            with col_btn:
                ref_id = str(s.get("ref_id") or "")
                if ref_id and ENTITY_INDEX.get(ref_id) is not None:
                    st.markdown('<div class="btn-pad"></div>', unsafe_allow_html=True)
                    st.button(
                        "地图",
                        key=f"loc_{d}_{idx}_{ref_id}",
                        help="在地图上定位这个地点",
                        on_click=_set_focus,
                        args=(d, ref_id),
                    )

    clar = state.get("clarifications") or []
    if clar:
        st.markdown(
            '<div class="note">需确认：' + html.escape("；".join(str(c) for c in clar)) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="note">照片为公开图文检索到的实景图，仅作目的地示意；'
        "门票与营业时间请以景区、餐厅当日公告为准。</div>",
        unsafe_allow_html=True,
    )


def render_retrieval(state: dict) -> None:
    trace = state.get("retrieval_trace", []) or []
    if not trace:
        st.markdown('<div class="note">无检索轨迹。</div>', unsafe_allow_html=True)
    else:
        rows = [
            {
                "来源": t.get("worker"),
                "数据源": t.get("source"),
                "状态": t.get("status"),
                "命中数": t.get("rows"),
                "耗时(ms)": t.get("latency_ms"),
                "错误": (t.get("error") or "")[:60],
            }
            for t in trace
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    for name, payload in (state.get("worker_results") or {}).items():
        if payload.get("error"):
            st.caption(f"{name}：{payload['error']}")
        attempts = payload.get("attempts") or []
        if attempts:
            with st.expander(f"Text2Cypher Fallback 链路（{name}，{len(attempts)} 步）",
                             expanded=False):
                st.dataframe(
                    [
                        {
                            "步骤": a.get("step"),
                            "轮次": a.get("attempt", "-"),
                            "状态": a.get("status"),
                            "错误": (a.get("error") or "")[:90],
                            "Cypher": (a.get("cypher") or "")[:90],
                        }
                        for a in attempts
                    ],
                    use_container_width=True, hide_index=True,
                )

    for name, payload in (state.get("worker_results") or {}).items():
        rows = payload.get("data") or []
        if name in ("poi", "dining") and isinstance(rows, list) and rows:
            with st.expander(f"候选地点与关联依据（{name}，前 8 条）", expanded=False):
                for r in rows[:8]:
                    st.markdown(
                        f"**{r.get('name', '')}** ｜ 关联层级 {r.get('graph_hops')} "
                        f"｜ 相关度 {r.get('graph_score')}"
                    )
                    for path in (r.get("graph_path") or []):
                        st.code(path, language="text")

    st.markdown('<div class="section-label">区域社区摘要</div>', unsafe_allow_html=True)
    area = st.selectbox("区域", AREAS, key="area_select", label_visibility="collapsed")
    s = area_summary(area)
    cells = "".join(
        f'<div class="kpi-item"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>'
        for label, value in [
            ("POI", s["poi_count"]),
            ("餐厅", s["restaurant_count"]),
            ("室内 POI", s["indoor_poi"]),
            ("无障碍 POI", s["wheelchair_poi"]),
            ("平均强度", s["avg_intensity"]),
        ]
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)


def render_check(state: dict) -> None:
    report = state.get("critic_report", {}) or {}
    viols = report.get("violations", []) or []
    fatal = int(report.get("fatal_count", 0) or 0)
    warn = int(report.get("warning_count", 0) or 0)

    cells = "".join(
        f'<div class="kpi-item"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {tone}">{value}</div></div>'
        for label, value, tone in [
            ("冲突提醒", fatal, "ok" if fatal == 0 else "danger"),
            ("温馨提示", warn, ""),
            ("结论", "通过" if report.get("passed") else "未通过", ""),
        ]
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)

    if not viols:
        st.markdown('<div class="note">已满足你提出的全部硬性条件。</div>', unsafe_allow_html=True)
    else:
        for v in viols:
            cls = "warn" if v.get("severity") != "fatal" else ""
            st.markdown(
                f'<div class="viol"><div class="viol-code {cls}">'
                f'{html.escape(str(v.get("code", "")))} · 第 {v.get("day", "-")} 天</div>'
                f'<div class="viol-detail">{html.escape(str(v.get("detail", "")))}</div>'
                f'<div class="viol-sug">建议：{html.escape(str(v.get("suggestion", "")))}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">自动调整记录</div>', unsafe_allow_html=True)
    history = state.get("replan_history", []) or []
    if not history:
        st.markdown('<div class="note">一次通过，无需调整。</div>', unsafe_allow_html=True)
        return
    hop_label = {"assembler": "重新编排", "planner": "重新排期"}
    for h in history:
        codes = "、".join(str(c) for c in (h.get("codes") or []))
        st.markdown(
            f'<div class="replan"><span class="round-chip">第 {h.get("round")} 轮</span>'
            f' → {html.escape(hop_label.get(str(h.get("next", "")), "重新编排"))}'
            f'<div class="replan-meta">冲突 {h.get("fatal_count")} 项 ｜ 类型 '
            f'{html.escape(codes) or "无"}</div></div>',
            unsafe_allow_html=True,
        )
        if h.get("excluded"):
            st.caption("本轮已替换：" + "、".join(str(x) for x in h["excluded"]))


def render_arch(state: dict) -> None:
    executed = {"router", "planner", "workers", "assembler", "critic", "finalize"}
    if state.get("replan_history"):
        executed.add("replan")
    st.markdown('<div class="panel">' + dag_html(executed) + "</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">已识别的需求条件</div>', unsafe_allow_html=True)
        st.json({k: v for k, v in (state.get("constraints") or {}).items()
                 if not str(k).startswith("_")})
    with col2:
        st.markdown('<div class="section-label">你的偏好记录</div>', unsafe_allow_html=True)
        st.json(state.get("user_profile") or {})


# 快速模式单个节点只有毫秒级，进度面板会「一步蹦完」看不出过程；
# 这里只给**展示节奏**加固定间隔：正常 6 个节点 → 5 次间隔 × 5s ≈ 25 秒。
# 只影响展示，不改变图执行；这些间隔会从界面显示的「生成耗时」里扣除。
FAST_STEP_DELAY = 5.0

# Assembler 帧内逐个揭示行程点位：每次只画前 k 个槽位，1 秒一个点。
# 仅快速模式生效；设为 0 则一帧画完全部点位。
LIVE_SLOT_DELAY = 1.0

# 过程地图高度（px）：太大会把下半部分推出首屏，改这里即可调整。
LIVE_MAP_HEIGHT = 360

STEP_LABEL = {
    "router": "识别需求与硬约束",
    "planner": "规划行程骨架",
    "workers": "并发检索景点 / 餐厅 / 天气",
    "assembler": "物化行程点位",
    "critic": "硬约束校验",
    "replan": "回跳修正",
    "finalize": "生成最终行程",
}


def step_hint(node: str, state: dict) -> str:
    """为进度行补一句该节点产出的结果摘要。"""
    if node == "router":
        c = state.get("constraints") or {}
        who = [name for name, key in (("老人", "with_elderly"), ("儿童", "with_kids"),
                                      ("无障碍", "wheelchair")) if c.get(key)]
        return f" · {c.get('days', 3)} 天 / {'+'.join(who) or '无特殊人群'}"
    if node == "planner":
        sk = (state.get("plan") or {}).get("skeleton") or []
        areas = "、".join(dict.fromkeys(s.get("area", "") for s in sk if s.get("area")))
        return f" · {len(sk)} 天骨架（{areas}）" if sk else ""
    if node == "workers":
        results = state.get("worker_results") or {}
        labels = {"poi": "景点", "dining": "餐厅", "weather": "天气"}
        parts = []
        for key in ("poi", "dining", "weather"):
            data = (results.get(key) or {}).get("data")
            parts.append(f"{labels[key]} {len(data) if isinstance(data, (list, dict)) else 0}")
        return " · " + " / ".join(parts)
    if node == "assembler":
        slots = (state.get("plan") or {}).get("slots") or []
        return f" · 已排 {len(slots)} 个点位"
    if node == "critic":
        report = state.get("critic_report") or {}
        if report.get("passed"):
            return " · 全部硬约束通过"
        return f" · {report.get('fatal_count', 0)} 项冲突，触发修正"
    if node == "replan":
        history = state.get("replan_history") or []
        if not history:
            return ""
        last = history[-1]
        return f" · 第 {last.get('round', 1)} 轮 → {last.get('next', 'assembler')}"
    if node == "finalize":
        return " · 行程已生成"
    return ""


# 编排过程地图：每完成一个节点，把当时的 State 快照投影到卫星底图上。
# 只画点与直线示意，不请求 OSRM 道路折线，避免拖慢编排节奏。
LIVE_STAGE = {
    "router": ("01 / 锚点", "锁定住宿与出行范围"),
    "planner": ("02 / 骨架", "按天分配区域"),
    "workers": ("03 / 召回", "候选景点与餐厅进入地图"),
    "assembler": ("04 / 物化", "从候选中挑出具体行程点位"),
    "critic": ("05 / 校验", "冲突点位在图上标红"),
    "replan": ("06 / 修正", "被替换的点位留在图上对照"),
    "finalize": ("07 / 完成", "行程路线成形"),
}

LIVE_LABEL_CSS = """
<style>
.live-area {transform:translate(-50%,-150%);background:rgba(251,250,248,.94);
  border:1px solid #111111;color:#111111;padding:2px 6px;white-space:nowrap;
  font:600 10px/1.4 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;}
.live-flag {transform:translate(-50%,-180%);background:#9B2C2C;color:#fff;
  padding:1px 5px;white-space:nowrap;font:700 10px/1.4 sans-serif;}
.live-drop {transform:translate(-50%,-50%);color:#9B2C2C;font:700 16px/1 sans-serif;}
</style>
"""


@lru_cache(maxsize=1)
def _area_centers() -> dict:
    """区域锚点：取该区域所有景点坐标的均值（WGS-84，lng/lat）。"""
    buckets = {}
    for poi in POIS:
        buckets.setdefault(poi.area, []).append((poi.lng, poi.lat))
    return {
        area: (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))
        for area, pts in buckets.items()
    }


def _live_xy(row: dict):
    """兼容 GraphRAG / 向量召回两种 row 结构，返回 GCJ-02 的 (x, y)。"""
    lng, lat = row.get("lng"), row.get("lat")
    if not isinstance(lng, (int, float)) or not isinstance(lat, (int, float)):
        ent = ENTITY_INDEX.get(str(row.get("id") or ""))
        if ent is None:
            return None
        lng, lat = ent.lng, ent.lat
    return _to_map(float(lng), float(lat))


def _live_layers(state: dict) -> dict:
    """把 State 快照拆成地图图层；坐标统一为 GCJ-02 的 (x, y)。"""
    constraints = state.get("constraints") or {}
    plan = state.get("plan") or {}
    results = state.get("worker_results") or {}
    report = state.get("critic_report") or {}
    history = state.get("replan_history") or []

    hotel = next((h for h in HOTELS if h.name == constraints.get("hotel_name")), None)
    hotel_xy = _to_map(hotel.lng, hotel.lat) if hotel is not None else None

    areas = []
    for sk in (plan.get("skeleton") or []):
        center = _area_centers().get(sk.get("area") or "")
        if center:
            areas.append((int(sk.get("day") or 0), str(sk.get("area") or ""), _to_map(*center)))

    candidates = []
    for key, kind in (("poi", "poi"), ("dining", "dining")):
        rows = (results.get(key) or {}).get("data") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                xy = _live_xy(row)
                if xy is not None:
                    candidates.append((kind, str(row.get("name") or ""), xy))

    by_day = {}
    for slot in (plan.get("slots") or []):
        ent = ENTITY_INDEX.get(str(slot.get("ref_id") or ""))
        if ent is None:
            continue
        by_day.setdefault(int(slot.get("day") or 1), []).append(
            (slot, _to_map(ent.lng, ent.lat))
        )
    days = [
        (day, sorted(by_day[day], key=lambda t: PERIOD_ORDER.get(t[0].get("period"), 9)))
        for day in sorted(by_day)
    ]

    flags = {
        str(v.get("ref_id")) for v in (report.get("violations") or [])
        if v.get("severity") == "fatal" and v.get("ref_id")
    }
    drops = {str(x) for x in (history[-1].get("excluded") or [])} if history else set()

    flag_xy, drop_xy = [], []
    for rid in flags | drops:
        ent = ENTITY_INDEX.get(rid)
        if ent is None:
            continue
        xy = _to_map(ent.lng, ent.lat)
        (flag_xy if rid in flags else drop_xy).append((ent.name, xy))

    xs, ys = [], []
    points = ([hotel_xy] if hotel_xy else []) + [a[2] for a in areas] \
        + [c[2] for c in candidates] + [p for _d, pts in days for _s, p in pts] \
        + [p for _n, p in flag_xy + drop_xy]
    for xy in points:
        xs.append(xy[0])
        ys.append(xy[1])

    return {
        "hotel": (hotel.name, hotel_xy) if hotel_xy else None,
        "areas": areas,
        "candidates": candidates,
        "days": days,
        "flags": flag_xy,
        "drops": drop_xy,
        "xs": xs,
        "ys": ys,
    }


def build_live_map(layers: dict, viewport: dict | None = None) -> folium.Map:
    """绘制一帧过程地图：候选为淡点，已排点位按天着色，冲突/替换单独标注。"""
    # 物化之后视野聚焦「酒店 + 行程点位」，不再被散布全市的候选池拉远；
    # viewport 传完整快照时，逐点揭示各帧共用同一视野，地图不会来回跳。
    base = viewport or layers
    focus = []
    if base["hotel"]:
        focus.append(base["hotel"][1])
    for _d, pts in base["days"]:
        focus.extend(xy for _s, xy in pts)
    focus.extend(xy for _n, xy in base["flags"] + base["drops"])
    view = focus if base["days"] else list(zip(base["xs"], base["ys"]))
    xs = [p[0] for p in view]
    ys = [p[1] for p in view]
    center = [(min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2] if xs else [22.55, 114.06]
    m = folium.Map(location=center, zoom_start=13 if len(xs) <= 1 else 12,
                   tiles=None, control_scale=False, prefer_canvas=True)
    m.options["scrollWheelZoom"] = False
    folium.TileLayer(tiles=AMAP_SAT, name="卫星影像", attr=AMAP_ATTR,
                     min_zoom=3, max_zoom=18, max_native_zoom=18).add_to(m)

    hotel = layers["hotel"]
    hotel_xy = hotel[1] if hotel else None

    # 行程示意路径：酒店 → 当天各点（直线，不调用 OSRM）
    for day, points in layers["days"]:
        color = _hex(DAY_COLORS[(day - 1) % len(DAY_COLORS)])
        line = ([hotel_xy] if hotel_xy else []) + [xy for _s, xy in points]
        if len(line) >= 2:
            folium.PolyLine(
                locations=[[y, x] for x, y in line], color=color,
                weight=2.2, opacity=0.8, dash_array="5,6",
            ).add_to(m)

    for kind, name, (x, y) in layers["candidates"]:
        color = "#C9A227" if kind == "poi" else "#7B5EA7"
        folium.CircleMarker(
            location=[y, x], radius=3.2, color=color, weight=0,
            fill=True, fill_color=color, fill_opacity=0.38,
            tooltip=f"候选 · {name}",
        ).add_to(m)

    for day, area, (x, y) in layers["areas"]:
        folium.CircleMarker(
            location=[y, x], radius=10, color="#111111", weight=1.2,
            fill=False, opacity=0.65,
        ).add_to(m)
        folium.Marker(
            location=[y, x],
            icon=folium.DivIcon(
                html=f'<div class="live-area">D{day} · {html.escape(area)}</div>',
                icon_size=(1, 1), icon_anchor=(0, 0),
            ),
        ).add_to(m)

    for day, points in layers["days"]:
        color = _hex(DAY_COLORS[(day - 1) % len(DAY_COLORS)])
        for i, (slot, (x, y)) in enumerate(points, 1):
            folium.CircleMarker(
                location=[y, x], radius=9.5, color="#ffffff", weight=2,
                fill=True, fill_color=color, fill_opacity=1,
                tooltip=f"第 {day} 天 · {slot.get('period', '')} · {slot.get('name', '')}",
            ).add_to(m)
            folium.Marker(
                location=[y, x],
                icon=folium.DivIcon(
                    html=f'<div class="map-num light">{i}</div>',
                    icon_size=(17, 17), icon_anchor=(8.5, 8.5),
                ),
            ).add_to(m)

    for _name, (x, y) in layers["flags"]:
        folium.Circle(location=[y, x], radius=170, color="#9B2C2C", weight=1.6,
                      fill=False).add_to(m)
        folium.Marker(
            location=[y, x],
            icon=folium.DivIcon(
                html='<div class="live-flag">冲突</div>',
                icon_size=(1, 1), icon_anchor=(0, 0),
            ),
        ).add_to(m)
    for name, (x, y) in layers["drops"]:
        folium.Marker(
            location=[y, x],
            icon=folium.DivIcon(
                html=f'<div class="live-drop" title="{html.escape(name)}">×</div>',
                icon_size=(14, 14), icon_anchor=(7, 7),
            ),
        ).add_to(m)

    if hotel:
        name, (x, y) = hotel
        folium.CircleMarker(
            location=[y, x], radius=10, color="#ffffff", weight=2,
            fill=True, fill_color="#545454", fill_opacity=1,
            tooltip=f"住宿 · {name}",
        ).add_to(m)
        folium.Marker(
            location=[y, x],
            icon=folium.DivIcon(
                html='<div class="map-num light">住</div>',
                icon_size=(17, 17), icon_anchor=(8.5, 8.5),
            ),
        ).add_to(m)

    if len(xs) > 1:
        m.fit_bounds([[min(ys), min(xs)], [max(ys), max(xs)]],
                     padding=(50, 50), max_zoom=14)
    m.get_root().html.add_child(folium.Element(MAP_INNER_CSS + LIVE_LABEL_CSS))
    return m


def render_live_map(placeholder, node: str, state: dict, seq: int | str,
                    reveal: tuple | None = None, viewport: dict | None = None) -> None:
    """在进度面板旁刷新一帧过程地图（用占位符替换上一帧，不叠加多张地图）。"""
    layers = _live_layers(state)
    title, sub = LIVE_STAGE.get(node, (node, ""))
    if reveal:
        sub = f"{sub} · 正在物化第 {reveal[0]} / {reveal[1]} 个点位"
    counts = [
        f"候选 {len(layers['candidates'])}",
        f"行程 {sum(len(pts) for _d, pts in layers['days'])}",
        f"冲突 {len(layers['flags'])}",
    ]
    with placeholder.container():
        st.markdown(
            '<div class="live-map-anchor"></div>'
            f'<div class="section-label">编排地图 · {html.escape(title)}</div>'
            f'<div class="map-caption-meta">{html.escape(sub)} · {" / ".join(counts)}</div>',
            unsafe_allow_html=True,
        )
        # 点位清单放在地图上方：即使地图下半部需要滚动，行程信息也始终可见
        stops = [(day, slot) for day, points in layers["days"] for slot, _xy in points]
        if stops:
            chips = []
            for i, (day, slot) in enumerate(stops, 1):
                current = " current" if reveal and i == reveal[0] else ""
                chips.append(
                    f'<span class="live-stop{current}"><b>D{day}</b>'
                    f'{html.escape(str(slot.get("period", "")))} · '
                    f'{html.escape(str(slot.get("name", "")))}</span>'
                )
            st.markdown(f'<div class="live-list">{"".join(chips)}</div>',
                        unsafe_allow_html=True)
        if not layers["xs"]:
            st.markdown('<div class="note">当前节点还没有可绘制的地理点位。</div>',
                        unsafe_allow_html=True)
            return
        st_folium(
            build_live_map(layers, viewport), height=LIVE_MAP_HEIGHT,
            use_container_width=True, returned_objects=[], key=f"live_map_{seq}",
        )
        st.markdown(
            '<div class="legend">'
            '<span class="legend-item"><span class="legend-dot" style="background:#C9A227"></span>候选景点</span>'
            '<span class="legend-item"><span class="legend-dot" style="background:#7B5EA7"></span>候选餐厅</span>'
            '<span class="legend-item"><span class="legend-dot" style="background:#E65100"></span>已排点位</span>'
            '<span class="legend-item"><span class="legend-dot" style="background:#9B2C2C"></span>冲突 / 替换</span>'
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="深圳家庭出行管家", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="masthead">'
        '<div class="kicker">Shenzhen · Family Travel</div>'
        "<h1>深圳家庭出行管家</h1></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="section-label">运行配置</div>', unsafe_allow_html=True)
        mode = st.radio(
            "生成方式",
            ["快速生成（秒级）", "精编生成（更细致）"],
            index=0,
            help="快速生成无需 API Key；精编会逐条推敲，耗时更长。",
        )
        enable_llm_critic = st.checkbox("启用更细致的语义复核", value=False)
        has_key = bool(get_api_key())
        if mode.startswith("精编") and not has_key:
            st.error("未检测到 API Key，请配置 apik.txt 或 OPENROUTER_API_KEY")

    st.markdown('<div class="section-label">01 / 出行需求</div>', unsafe_allow_html=True)
    ex_cols = st.columns(4)
    for i, (label, text) in enumerate(EXAMPLES):
        ex_cols[i].button(
            label, key=f"ex{i}", on_click=_fill_example, args=(text,),
            use_container_width=True, help=text,
        )

    with st.form("trip_form", clear_on_submit=False):
        st.text_area(
            "出行需求",
            key="query_text",
            height=100,
            label_visibility="collapsed",
            placeholder="例如：带 70 岁老人和 6 岁小孩去深圳玩 3 天，需要轮椅无障碍，"
                        "饮食清淡，住福田香格里拉……",
        )
        col_btn, col_hint = st.columns([1, 2])
        submitted = col_btn.form_submit_button("生成行程", type="primary", use_container_width=True)
        col_hint.caption("快速生成 1–3 秒；精编约 1–9 分钟（需 API Key）")

    if submitted:
        q = (st.session_state.get("query_text") or "").strip()
        if not q:
            st.warning("请输入出行需求")
        else:
            os.environ["FAST_MODE"] = "0" if mode.startswith("精编") else "1"
            os.environ["ENABLE_LLM_CRITIC"] = "1" if enable_llm_critic else "0"
            started = time.time()
            trip_state = None
            graph_elapsed = 0.0
            display_delay = 0.0
            fast_run = not mode.startswith("精编")
            col_run, col_live = st.columns([1, 1.6], gap="large")
            live_placeholder = col_live.empty()
            with col_run:
                with st.status("正在编排行程…", expanded=True) as run_status:
                    try:
                        first_step = True
                        for seq, (node, snapshot, step_elapsed) in enumerate(
                                stream_trip(q, session_id="web")):
                            # 仅快速模式放慢展示节奏：首行立即出现，其后每步留出可读间隔
                            if fast_run and not first_step and FAST_STEP_DELAY > 0:
                                time.sleep(FAST_STEP_DELAY)
                                display_delay += FAST_STEP_DELAY
                            first_step = False
                            run_status.write(
                                f"`{step_elapsed:>6.2f}s` **{STEP_LABEL.get(node, node)}**"
                                f"{step_hint(node, snapshot)}"
                            )
                            # Assembler 帧逐点揭示，其余节点一帧画完
                            try:
                                slots = (snapshot.get("plan") or {}).get("slots") or []
                                if node == "assembler" and fast_run and slots:
                                    total = len(slots)
                                    full_layers = _live_layers(snapshot)
                                    for k in range(1, total + 1):
                                        partial = {
                                            **snapshot,
                                            "plan": {**(snapshot.get("plan") or {}),
                                                     "slots": slots[:k]},
                                        }
                                        render_live_map(live_placeholder, node, partial,
                                                        f"{seq}_{k}", reveal=(k, total),
                                                        viewport=full_layers)
                                        if k < total and LIVE_SLOT_DELAY > 0:
                                            time.sleep(LIVE_SLOT_DELAY)
                                            display_delay += LIVE_SLOT_DELAY
                                else:
                                    render_live_map(live_placeholder, node, snapshot, seq)
                            except Exception as map_exc:  # noqa: BLE001
                                live_placeholder.caption(
                                    f"过程地图暂不可用：{type(map_exc).__name__}: {map_exc}")
                            trip_state = snapshot
                            graph_elapsed = step_elapsed
                    except Exception as exc:  # noqa: BLE001
                        run_status.update(label="编排失败", state="error", expanded=False)
                        st.error(f"运行失败：{type(exc).__name__}: {exc}")
                        st.stop()
                    # 扣除展示节奏引入的 sleep，还原真实的图执行耗时
                    elapsed = max(0.0, (graph_elapsed or (time.time() - started)) - display_delay)
                    run_status.update(label=f"编排完成 · {elapsed:.1f}s",
                                      state="complete", expanded=False)
            # 结果区自带完整道路路线地图，编排过程地图完成后让位
            live_placeholder.empty()
            if not trip_state:
                st.error("运行失败：未产生有效行程状态")
                st.stop()
            st.session_state["last_state"] = trip_state
            st.session_state["last_elapsed"] = elapsed
            st.session_state["last_query"] = q

    state = st.session_state.get("last_state")
    if not state:
        st.markdown(
            '<div class="note">输入需求或点选一个示例，然后点击「生成行程」开始。</div>',
            unsafe_allow_html=True,
        )
        return

    elapsed = float(st.session_state.get("last_elapsed") or 0.0)
    render_kpis(state, elapsed)

    tab_plan, tab_retrieval, tab_check, tab_arch = st.tabs(
        ["行程", "推荐依据", "校验记录", "编排说明"]
    )

    with tab_plan:
        col_plan, col_map = st.columns([1.45, 1], gap="large")
        with col_plan:
            col_dl, _ = st.columns([1, 2])
            col_dl.download_button(
                "导出 Markdown",
                data=state.get("answer", ""),
                file_name="深圳家庭游行程.md",
                mime="text/markdown",
                use_container_width=True,
            )
            render_itinerary(state)
        with col_map:
            render_map_panel(state)

    with tab_retrieval:
        render_retrieval(state)

    with tab_check:
        render_check(state)

    with tab_arch:
        render_arch(state)


if __name__ == "__main__":
    main()
