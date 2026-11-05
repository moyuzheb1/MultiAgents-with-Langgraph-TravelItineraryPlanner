"""家庭出行管家 · Supervisor-Worker 多智能体系统（Streamlit 可视化控制台）。

界面为极简杂志风：
- 报头 + 细分隔线 + 大号衬线标题
- 无圆角、无阴影、无 emoji
- 顶部 KPI 数据带 + 路线平面地图 + 按天编排的行程版式
- 检索依据 / 校验回跳 / 架构记忆 分栏呈现
"""
from __future__ import annotations

import html
import math
import os
import time

import pydeck as pdk
import streamlit as st

from agents import run_trip  # noqa: E402
from core.llm import DEFAULT_MODEL, get_api_key  # noqa: E402
from core.rules import ENTITY_INDEX  # noqa: E402
from data.shenzhen import HOTELS  # noqa: E402
from retrieval import area_summary, get_cache, get_graph_store, get_vector_store  # noqa: E402

EXAMPLES = [
    ("老年无障碍", "带70岁老人和6岁小孩去深圳玩3天，需要轮椅无障碍，饮食清淡，"
                 "住深圳福田香格里拉大酒店，避开海鲜排档"),
    ("台风天备选", "深圳2日游，遇到台风，带老人，要避开海鲜排档"),
    ("低龄亲子", "一家人去深圳玩4天，小孩3岁，人均餐标100以内"),
    ("大梅沙休闲", "带老人去大梅沙附近玩2天，不要爬山，雨天要有室内备选"),
]

AREAS = ["南山区", "福田区", "罗湖区", "盐田区", "龙岗区", "宝安区", "龙华区", "光明区", "大鹏新区"]

PERIOD_ORDER = {"上午": 1, "午餐": 2, "下午": 3, "晚餐": 4, "晚上": 5}
PERIOD_TIME = {"上午": "09:00", "午餐": "12:00", "下午": "15:00", "晚餐": "18:00", "晚上": "20:00"}
KIND_META = {"poi": ("景点", "poi"), "dining": ("餐饮", "dining"), "rest": ("休息", "rest")}

DAY_COLORS = [
    [17, 17, 17],
    [155, 44, 44],
    [47, 93, 80],
    [107, 91, 62],
    [60, 70, 110],
]
KIND_COLORS = {"poi": [17, 17, 17], "dining": [155, 44, 44], "hotel": [107, 107, 107]}

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
.masthead-sub {font-size: 13px; color: var(--muted); max-width: 820px; line-height: 1.75;}
.meta-row {
  display: flex; gap: 22px; flex-wrap: wrap; font-size: 10px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--muted);
  border-bottom: 1px solid var(--line); padding: 10px 0 12px; margin-bottom: 22px;
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
.legend-note {margin-left: auto; font-size: 10px; letter-spacing: .06em;}

.day {display: grid; grid-template-columns: 92px 1fr; gap: 24px; padding: 22px 0; border-top: 1px solid var(--line);}
.day:first-of-type {border-top: 1px solid var(--ink);}
.day-index {font-family: Georgia, "Times New Roman", serif; font-size: 40px; line-height: 1; color: var(--accent);}
.day-kicker {font-size: 10px; letter-spacing: .2em; text-transform: uppercase; color: var(--muted);}
.day-title {
  font-family: "Songti SC", "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
  font-size: 22px; margin: 4px 0 10px;
}
.stop {display: grid; grid-template-columns: 54px 1fr auto; gap: 14px; padding: 10px 0; border-top: 1px dotted var(--line);}
.day-body > .stop:first-of-type {border-top: none;}
.stop-time {font-size: 11px; letter-spacing: .06em; color: var(--muted); padding-top: 3px;}
.stop-name {font-size: 15px; font-weight: 600;}
.stop-note {font-size: 12px; color: var(--muted); margin-top: 2px; line-height: 1.6;}
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
  .day {grid-template-columns: 1fr; gap: 8px;}
  .day-index {font-size: 28px;}
  .stop {grid-template-columns: 46px 1fr;}
  .stop-kind {display: none;}
}
</style>
"""


def backend_status() -> str:
    g = get_graph_store().backend
    v = get_vector_store().backend
    c = get_cache().backend
    return f"图存储 {g} ｜ 向量 {v} ｜ 缓存 {c}"


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
        box("router", "Router", "意图 / 约束"), arrow(),
        box("planner", "Planner", "行程骨架"), arrow(),
        box("workers", "Workers", "图 / 向量 / 天气"), arrow(),
        box("assembler", "Assembler", "确定性物化"), arrow(),
        box("critic", "Critic", "硬约束校验"),
    ])
    tail = (
        f'{arrow("通过", "#2F5D50")}{box("finalize", "Finalize", "收尾输出")}'
        f'<div style="margin-top:14px;color:#9B2C2C;font-size:11px">'
        f'↺ 致命违规 → Replan（局部 State 更新）→ 回跳 Assembler / Planner</div>'
    )
    return (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'font-family:sans-serif">{chain}{tail}</div>'
        f'<div style="margin-top:16px">{box("replan", "Replan", "局部更新")} '
        f'<span style="color:#9B2C2C;font-size:11px">条件边，按需插入循环</span></div>'
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
        ("行程安排", str(len(slots)), "个槽位", ""),
        ("致命违规", str(fatal), "硬约束拦截", "ok" if fatal == 0 else "danger"),
        ("提示项", str(warn), "非阻断提醒", ""),
        ("耗时", f"{elapsed:.1f}", "秒", ""),
    ]
    cells = "".join(
        f'<div class="kpi-item"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value {tone}">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div></div>'
        for label, value, sub, tone in items
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)


def _build_map(state: dict):
    plan = state.get("plan", {}) or {}
    slots = plan.get("slots", []) or []
    constraints = state.get("constraints", {}) or {}
    days = int(plan.get("days") or constraints.get("days") or 3)
    hotel = next((h for h in HOTELS if h.name == constraints.get("hotel_name")), None)

    routes, stops = [], []
    lngs, lats = [], []

    for d in range(1, days + 1):
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
        path = []
        if hotel is not None:
            path.append([hotel.lng, hotel.lat])
            lngs.append(hotel.lng)
            lats.append(hotel.lat)
        for _kind, ent in seq:
            path.append([ent.lng, ent.lat])
            lngs.append(ent.lng)
            lats.append(ent.lat)
        if hotel is not None:
            path.append([hotel.lng, hotel.lat])

        routes.append({"day": d, "path": path, "color": color})
        for i, (kind, ent) in enumerate(seq, 1):
            stops.append({
                "day": d,
                "position": [ent.lng, ent.lat],
                "label": str(i),
                "name": ent.name,
                "kind": kind,
                "color": KIND_COLORS.get(kind, KIND_COLORS["poi"]),
            })

    if hotel is not None:
        stops.append({
            "day": 0,
            "position": [hotel.lng, hotel.lat],
            "label": "住",
            "name": hotel.name,
            "kind": "hotel",
            "color": KIND_COLORS["hotel"],
        })
        lngs.append(hotel.lng)
        lats.append(hotel.lat)

    return routes, stops, lngs, lats


def _view_state(lngs: list, lats: list) -> pdk.ViewState:
    if not lngs or not lats:
        return pdk.ViewState(latitude=22.55, longitude=114.05, zoom=10)
    lat0 = (min(lats) + max(lats)) / 2
    lng0 = (min(lngs) + max(lngs)) / 2
    span = max(max(lats) - min(lats), max(lngs) - min(lngs), 0.004)
    zoom = 13.2 - math.log2(span / 0.02)
    zoom = max(9.5, min(13.0, zoom))
    return pdk.ViewState(latitude=lat0, longitude=lng0, zoom=zoom, pitch=0, bearing=0)


def render_map(state: dict) -> None:
    routes, stops, lngs, lats = _build_map(state)
    st.markdown('<div class="map-caption">路线平面图</div>', unsafe_allow_html=True)
    if not routes or not stops:
        st.markdown('<div class="note">当前行程暂无可绘制的坐标点。</div>', unsafe_allow_html=True)
        return

    layers = [
        pdk.Layer(
            "TileLayer",
            data="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
        ),
        pdk.Layer(
            "PathLayer",
            data=routes,
            get_path="path",
            get_color="color",
            get_width=3,
            width_min_pixels=2,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=stops,
            get_position="position",
            get_fill_color="color",
            get_radius=110,
            radius_min_pixels=6,
            radius_max_pixels=16,
            pickable=True,
        ),
        pdk.Layer(
            "TextLayer",
            data=stops,
            get_position="position",
            get_text="label",
            get_size=12,
            get_color=[255, 255, 255],
            get_pixel_offset=[0, 0],
        ),
    ]

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=_view_state(lngs, lats),
        map_style=None,
        tooltip={"text": "{name}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    legend = "".join(
        f'<span class="legend-item"><span class="legend-swatch" '
        f'style="background:rgb({c[0]},{c[1]},{c[2]})"></span>第 {i + 1} 天</span>'
        for i, c in enumerate(DAY_COLORS[:len(routes)])
    )
    st.markdown(
        f'<div class="legend">{legend}'
        f'<span class="legend-note">数字为当日顺序 · 底图 CARTO / OpenStreetMap</span></div>',
        unsafe_allow_html=True,
    )


def _slot_html(slot: dict) -> str:
    period = slot.get("period", "")
    kind = slot.get("kind", "")
    label, cls = KIND_META.get(kind, (kind or "安排", "other"))
    return (
        f'<div class="stop">'
        f'<div class="stop-time">{PERIOD_TIME.get(period, "--:--")}</div>'
        f'<div><div class="stop-name">{html.escape(str(slot.get("name", "")))}</div>'
        f'<div class="stop-note">{html.escape(str(slot.get("note", "")))}</div></div>'
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

        body = "".join(_slot_html(s) for s in day_slots)
        if not day_slots:
            body = (
                '<div class="stop"><div class="stop-time">--:--</div>'
                '<div><div class="stop-name">自由活动 / 酒店休息</div>'
                '<div class="stop-note">当日无满足硬约束的安排</div></div>'
                '<div class="stop-kind rest">休息</div></div>'
            )

        st.markdown(
            f'<div class="day"><div class="day-index">{d:02d}</div>'
            f'<div class="day-body">'
            f'<div class="day-kicker">Day {d}{" · " + html.escape(meta) if meta else ""}</div>'
            f'<div class="day-title">{html.escape(title)}</div>{body}</div></div>',
            unsafe_allow_html=True,
        )

    clar = state.get("clarifications") or []
    if clar:
        st.markdown(
            '<div class="note">需确认：' + html.escape("；".join(str(c) for c in clar)) + "</div>",
            unsafe_allow_html=True,
        )


def render_retrieval(state: dict) -> None:
    trace = state.get("retrieval_trace", []) or []
    if not trace:
        st.markdown('<div class="note">无检索轨迹。</div>', unsafe_allow_html=True)
    else:
        rows = [
            {
                "Worker": t.get("worker"),
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

    for name, payload in (state.get("worker_results") or {}).items():
        rows = payload.get("data") or []
        if name in ("poi", "dining") and isinstance(rows, list) and rows:
            with st.expander(f"GraphRAG 多跳路径（{name}，前 8 条）", expanded=False):
                for r in rows[:8]:
                    st.markdown(
                        f"**{r.get('name', '')}** ｜ hops `{r.get('graph_hops')}` "
                        f"｜ score `{r.get('graph_score')}`"
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
            ("致命违规", fatal, "ok" if fatal == 0 else "danger"),
            ("提示项", warn, ""),
            ("结论", "通过" if report.get("passed") else "未通过", ""),
        ]
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)

    if not viols:
        st.markdown('<div class="note">全部硬约束校验通过。</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-label">Replan 回跳时间线</div>', unsafe_allow_html=True)
    history = state.get("replan_history", []) or []
    if not history:
        st.markdown('<div class="note">一次通过，无需重规划。</div>', unsafe_allow_html=True)
        return
    for h in history:
        codes = "、".join(str(c) for c in (h.get("codes") or []))
        st.markdown(
            f'<div class="replan"><span class="round-chip">第 {h.get("round")} 轮</span>'
            f' → {html.escape(str(h.get("next", "")))}'
            f'<div class="replan-meta">致命违规 {h.get("fatal_count")} 条 ｜ 违规码 '
            f'{html.escape(codes) or "无"}</div></div>',
            unsafe_allow_html=True,
        )
        if h.get("excluded"):
            st.caption("排除实体：" + "、".join(str(x) for x in h["excluded"]))


def render_arch(state: dict) -> None:
    executed = {"router", "planner", "workers", "assembler", "critic", "finalize"}
    if state.get("replan_history"):
        executed.add("replan")
    st.markdown('<div class="panel">' + dag_html(executed) + "</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">抽取硬约束</div>', unsafe_allow_html=True)
        st.json({k: v for k, v in (state.get("constraints") or {}).items()
                 if not str(k).startswith("_")})
    with col2:
        st.markdown('<div class="section-label">用户画像（长期记忆）</div>', unsafe_allow_html=True)
        st.json(state.get("user_profile") or {})


def main() -> None:
    st.set_page_config(page_title="家庭出行管家", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="masthead">'
        '<div class="kicker">Multi-Agent Travel Butler</div>'
        "<h1>家庭出行管家</h1>"
        '<div class="masthead-sub">Supervisor–Worker 多智能体编排：Router → Planner → '
        '并发 Workers → Assembler → Critic ⇄ Replan。</div></div>'
        '<div class="meta-row"><span>GraphRAG 多跳检索</span>'
        '<span>Text2Cypher 防御性纠错</span><span>确定性物化</span>'
        '<span>硬约束自愈回跳</span></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="section-label">运行配置</div>', unsafe_allow_html=True)
        mode = st.radio(
            "运行模式",
            ["快速模式（确定性）", "智能体增强（LLM）"],
            index=0,
            help="快速模式秒级、无需 API Key；增强模式调用 LLM，较慢但更智能。",
        )
        enable_llm_critic = st.checkbox("启用 LLM 语义复核（Critic）", value=False)
        st.markdown('<div class="section-label">当前后端</div>', unsafe_allow_html=True)
        st.markdown(backend_status())
        st.markdown(f"模型：`{DEFAULT_MODEL}`")
        has_key = bool(get_api_key())
        if mode.startswith("智能体增强") and not has_key:
            st.error("未检测到 API Key，请配置 apik.txt 或 OPENROUTER_API_KEY")
        with st.expander("架构说明"):
            st.markdown(
                "1. Router：意图识别 + 硬约束抽取\n"
                "2. Planner：只生成骨架，规避幻觉\n"
                "3. Workers：GraphRAG / 向量 / 天气三路并发\n"
                "4. Assembler：确定性物化，杜绝幻觉点位\n"
                "5. Critic：规则校验 + 可选 LLM 语义复核\n"
                "6. Replan：拦截 → 触发 → 局部 State 更新 → 回跳"
            )

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
        col_hint.caption("快速模式 1–3 秒；智能体增强约 1–9 分钟（需 API Key）")

    if submitted:
        q = (st.session_state.get("query_text") or "").strip()
        if not q:
            st.warning("请输入出行需求")
        else:
            os.environ["FAST_MODE"] = "0" if mode.startswith("智能体增强") else "1"
            os.environ["ENABLE_LLM_CRITIC"] = "1" if enable_llm_critic else "0"
            started = time.time()
            with st.spinner("正在编排行程…"):
                try:
                    trip_state = run_trip(q, session_id="web")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"运行失败：{type(exc).__name__}: {exc}")
                    st.stop()
            elapsed = time.time() - started
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
    render_map(state)

    tab_plan, tab_retrieval, tab_check, tab_arch = st.tabs(
        ["行程", "检索依据", "校验与回跳", "架构与记忆"]
    )

    with tab_plan:
        col_dl, _ = st.columns([1, 3])
        col_dl.download_button(
            "导出 Markdown",
            data=state.get("answer", ""),
            file_name="深圳家庭游行程.md",
            mime="text/markdown",
            use_container_width=True,
        )
        render_itinerary(state)

    with tab_retrieval:
        render_retrieval(state)

    with tab_check:
        render_check(state)

    with tab_arch:
        render_arch(state)


if __name__ == "__main__":
    main()
