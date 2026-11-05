"""家庭出行管家 · Supervisor-Worker 多智能体系统（Streamlit 可视化控制台）。

界面采用「控制台骨架 + 行程卡片」：
- 顶部 Hero 与 KPI 指标行
- 左侧运行配置，右侧主结果
- 行程按天渲染为时间轴卡片
- 检索依据 / 校验回跳 / 架构记忆 分四个 Tab
"""
from __future__ import annotations

import html
import os
import time

import streamlit as st

from agents import run_trip  # noqa: E402
from core.llm import DEFAULT_MODEL, get_api_key  # noqa: E402
from retrieval import area_summary, get_cache, get_graph_store, get_vector_store  # noqa: E402

EXAMPLES = [
    (
        "👴 老人 + 无障碍",
        "带70岁老人和6岁小孩去深圳玩3天，需要轮椅无障碍，饮食清淡，"
        "住深圳福田香格里拉大酒店，避开海鲜排档",
    ),
    ("🌀 台风天备选", "深圳2日游，遇到台风，带老人，要避开海鲜排档"),
    ("👶 低龄亲子", "一家人去深圳玩4天，小孩3岁，人均餐标100以内"),
    ("🏖️ 大梅沙休闲", "带老人去大梅沙附近玩2天，不要爬山，雨天要有室内备选"),
]

AREAS = ["南山区", "福田区", "罗湖区", "盐田区", "龙岗区", "宝安区", "龙华区", "光明区", "大鹏新区"]

PERIOD_ORDER = {"上午": 1, "午餐": 2, "下午": 3, "晚餐": 4, "晚上": 5}
PERIOD_TIME = {"上午": "09:00", "午餐": "12:00", "下午": "15:00", "晚餐": "18:00", "晚上": "20:00"}
KIND_META = {"poi": ("景点", "poi"), "dining": ("餐饮", "dining"), "rest": ("休息", "rest")}

CSS = """
<style>
:root {
  --brand:#2563EB; --brand-dark:#1D4ED8; --ink:#0F172A; --muted:#64748B;
  --line:#E2E8F0; --card:#FFFFFF;
}
.block-container {max-width: 1200px; padding-top: 1.4rem; padding-bottom: 3rem;}
#MainMenu, footer {visibility: hidden;}

.hero {
  background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #2563EB 100%);
  color:#fff; padding:28px 32px; border-radius:20px; margin-bottom:18px;
  box-shadow:0 10px 30px rgba(37,99,235,.18);
}
.hero h1 {font-size:30px; margin:0 0 8px; font-weight:800; letter-spacing:.5px;}
.hero p {margin:0; color:#DBEAFE; font-size:14px; line-height:1.7;}
.hero .badges {margin-top:14px; display:flex; gap:8px; flex-wrap:wrap;}
.hero .badge {
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
  color:#EFF6FF; padding:4px 11px; border-radius:999px; font-size:12px;
}

.section-title {font-size:14px; font-weight:700; color:#334155; margin:6px 0 10px;}

.kpi {
  background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:14px 16px; box-shadow:0 2px 10px rgba(15,23,42,.04); height:100%;
}
.kpi-label {color:var(--muted); font-size:12px;}
.kpi-value {color:var(--ink); font-size:26px; font-weight:800; line-height:1.2; margin-top:2px;}
.kpi-sub {color:var(--muted); font-size:11px; margin-top:2px;}
.kpi.ok .kpi-value {color:#059669;}
.kpi.danger .kpi-value {color:#DC2626;}

.day-card {
  background:var(--card); border:1px solid var(--line); border-left:5px solid var(--brand);
  border-radius:16px; padding:18px 20px; margin-bottom:16px;
  box-shadow:0 3px 14px rgba(15,23,42,.05);
}
.day-head {display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:12px;}
.day-num {font-size:13px; font-weight:800; color:var(--brand); letter-spacing:1px;}
.day-title {font-size:17px; font-weight:700; color:var(--ink);}
.day-meta {margin-left:auto; color:var(--muted); font-size:12px;}
.slot {display:flex; gap:12px; align-items:flex-start; padding:9px 0; border-top:1px dashed var(--line);}
.day-head + .slot {border-top:none;}
.slot-time {width:52px; color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums; padding-top:2px;}
.slot-main {flex:1;}
.slot-name {color:var(--ink); font-weight:600; font-size:14px;}
.slot-note {color:var(--muted); font-size:12px; margin-top:1px;}
.tag {font-size:11px; padding:2px 9px; border-radius:999px; white-space:nowrap;}
.tag.poi {background:#EFF6FF; color:#1D4ED8;}
.tag.dining {background:#FFF7ED; color:#C2410C;}
.tag.rest {background:#ECFDF5; color:#047857;}
.tag.other {background:#F1F5F9; color:#475569;}

.viol {
  background:var(--card); border:1px solid var(--line); border-left:4px solid #DC2626;
  border-radius:12px; padding:12px 14px; margin-bottom:10px;
}
.viol.warn {border-left-color:#F59E0B;}
.viol-code {font-family:ui-monospace, Consolas, monospace; font-size:11px; color:var(--muted);}
.viol-detail {font-size:13px; color:var(--ink); margin-top:3px;}
.viol-sug {font-size:12px; color:var(--muted); margin-top:3px;}

.replan {
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:12px 14px; margin-bottom:10px;
}
.round-chip {
  background:#EFF6FF; color:var(--brand-dark); font-weight:700; font-size:12px;
  padding:2px 10px; border-radius:999px;
}

.panel {
  background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:18px 20px; margin-bottom:16px;
}
.empty {
  background:var(--card); border:1px dashed var(--line); border-radius:16px;
  padding:38px; text-align:center; color:var(--muted);
}
</style>
"""


def backend_status() -> str:
    g = get_graph_store().backend
    v = get_vector_store().backend
    c = get_cache().backend
    return f"图存储 **{g}** ｜ 向量 **{v}** ｜ 缓存 **{c}**"


def dag_html(executed: set) -> str:
    """用纯 HTML 渲染架构 DAG（避免 graphviz 二进制依赖）。"""

    def box(nid: str, title: str, sub: str) -> str:
        active = (
            "background:#EFF6FF;border-color:#2563EB;color:#1D4ED8"
            if nid in executed
            else "background:#F8FAFC;border-color:#CBD5E1;color:#94A3B8"
        )
        return (
            f'<div style="border:2px solid;{active};border-radius:10px;'
            f'padding:8px 12px;text-align:center;min-width:98px;font-size:12px">'
            f'<b>{title}</b><br><span style="font-size:11px">{sub}</span></div>'
        )

    def arrow(label: str = "", color: str = "#64748B") -> str:
        return (
            f'<div style="align-self:center;color:{color};font-size:12px;'
            f'margin:0 6px;text-align:center">→<br><span style="font-size:10px">{label}</span></div>'
        )

    chain = "".join([
        box("router", "Router", "意图/约束"), arrow(),
        box("planner", "Planner", "行程骨架"), arrow(),
        box("workers", "并发 Workers", "图/向量/天气"), arrow(),
        box("assembler", "Assembler", "确定性物化"), arrow(),
        box("critic", "Critic", "硬约束校验"),
    ])
    tail = (
        f'{arrow("通过", "#059669")}{box("finalize", "Finalize", "收尾输出")}'
        f'<div style="margin-top:14px;color:#DC2626;font-size:12px">'
        f'↺ Critic 检出致命违规 → <b>Replan</b>（局部 State 更新）'
        f'→ 回跳 Assembler / Planner</div>'
    )
    replan_box = box("replan", "Replan", "局部 State 更新")
    return (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
        f'font-family:sans-serif">{chain}{tail}</div>'
        f'<div style="margin-top:18px">{replan_box} '
        f'<span style="color:#DC2626;font-size:12px">（条件边，按需插入循环）</span></div>'
    )


def _fill_example(text: str) -> None:
    st.session_state["query_text"] = text


def _kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (
        f'<div class="kpi {tone}"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div></div>'
    )


def _slot_html(slot: dict) -> str:
    period = slot.get("period", "")
    kind = slot.get("kind", "")
    label, cls = KIND_META.get(kind, (kind or "安排", "other"))
    return (
        f'<div class="slot">'
        f'<div class="slot-time">{PERIOD_TIME.get(period, "--:--")}</div>'
        f'<div class="slot-main">'
        f'<div class="slot-name">{html.escape(str(slot.get("name", "")))}</div>'
        f'<div class="slot-note">{html.escape(str(slot.get("note", "")))}</div>'
        f'</div><span class="tag {cls}">{label}</span></div>'
    )


def render_kpis(state: dict, elapsed: float) -> None:
    plan = state.get("plan", {}) or {}
    slots = plan.get("slots", []) or []
    report = state.get("critic_report", {}) or {}
    constraints = state.get("constraints", {}) or {}
    days = int(plan.get("days") or constraints.get("days") or 0)
    fatal = int(report.get("fatal_count", 0) or 0)
    warn = int(report.get("warning_count", 0) or 0)

    cards = [
        ("行程天数", str(days), "天", ""),
        ("行程安排", str(len(slots)), "个槽位", ""),
        ("致命违规", str(fatal), "硬约束拦截", "ok" if fatal == 0 else "danger"),
        ("提示项", str(warn), "非阻断提醒", ""),
        ("耗时", f"{elapsed:.1f}", "秒", ""),
    ]
    cols = st.columns(len(cards))
    for col, (label, value, sub, tone) in zip(cols, cards):
        col.markdown(_kpi_card(label, value, sub, tone), unsafe_allow_html=True)


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
        title = " · ".join(x for x in (area, theme) if x) or f"第 {d} 天"
        w = weather.get(str(d), {}) or {}
        meta = " ｜ ".join(x for x in [w.get("level", ""), w.get("temp", "")] if x)

        body = "".join(_slot_html(s) for s in day_slots)
        if not day_slots:
            body = (
                '<div class="slot"><div class="slot-time">--:--</div>'
                '<div class="slot-main"><div class="slot-name">自由活动 / 酒店休息</div>'
                '<div class="slot-note">当日无满足硬约束的安排</div></div>'
                '<span class="tag rest">休息</span></div>'
            )

        st.markdown(
            f'<div class="day-card">'
            f'<div class="day-head">'
            f'<div class="day-num">DAY {d:02d}</div>'
            f'<div class="day-title">{html.escape(title)}</div>'
            f'<div class="day-meta">{html.escape(meta)}</div>'
            f'</div>{body}</div>',
            unsafe_allow_html=True,
        )

    clar = state.get("clarifications") or []
    if clar:
        st.warning("需确认：" + "；".join(str(c) for c in clar))


def render_retrieval(state: dict) -> None:
    trace = state.get("retrieval_trace", []) or []
    if not trace:
        st.info("无检索轨迹")
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
            with st.expander(f"🕸️ {name} · GraphRAG 多跳路径（前 8 条）", expanded=False):
                for r in rows[:8]:
                    st.markdown(
                        f"**{r.get('name', '')}** ｜ hops `{r.get('graph_hops')}` "
                        f"｜ score `{r.get('graph_score')}`"
                    )
                    for path in (r.get("graph_path") or []):
                        st.code(path, language="text")

    st.markdown('<div class="section-title">区域社区摘要（GraphRAG area_summary）</div>',
                unsafe_allow_html=True)
    area = st.selectbox("选择区域", AREAS, key="area_select", label_visibility="collapsed")
    s = area_summary(area)
    cols = st.columns(5)
    cols[0].metric("POI", s["poi_count"])
    cols[1].metric("餐厅", s["restaurant_count"])
    cols[2].metric("室内 POI", s["indoor_poi"])
    cols[3].metric("无障碍 POI", s["wheelchair_poi"])
    cols[4].metric("平均强度", s["avg_intensity"])


def render_check(state: dict) -> None:
    report = state.get("critic_report", {}) or {}
    viols = report.get("violations", []) or []
    fatal = int(report.get("fatal_count", 0) or 0)
    warn = int(report.get("warning_count", 0) or 0)

    cols = st.columns(3)
    cols[0].metric("致命违规", fatal)
    cols[1].metric("提示项", warn)
    cols[2].metric("结论", "通过" if report.get("passed") else "未通过")

    if not viols:
        st.success("全部硬约束校验通过")
    else:
        for v in viols:
            cls = "" if v.get("severity") == "fatal" else "warn"
            st.markdown(
                f'<div class="viol {cls}">'
                f'<div class="viol-code">{html.escape(str(v.get("code", "")))}'
                f' · 第 {v.get("day", "-")} 天</div>'
                f'<div class="viol-detail">{html.escape(str(v.get("detail", "")))}</div>'
                f'<div class="viol-sug">建议：{html.escape(str(v.get("suggestion", "")))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Replan 回跳时间线</div>', unsafe_allow_html=True)
    history = state.get("replan_history", []) or []
    if not history:
        st.caption("一次通过，无需重规划。")
        return
    for h in history:
        codes = "、".join(str(c) for c in (h.get("codes") or []))
        st.markdown(
            f'<div class="replan"><span class="round-chip">第 {h.get("round")} 轮</span>'
            f' <b>→ {html.escape(str(h.get("next", "")))}</b>'
            f'<span style="color:#64748B;font-size:12px"> ｜ 致命违规 '
            f'{h.get("fatal_count")} 条 ｜ 违规码 {html.escape(codes)}</span></div>',
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
        st.markdown("**抽取硬约束**")
        st.json({k: v for k, v in (state.get("constraints") or {}).items()
                 if not str(k).startswith("_")})
    with col2:
        st.markdown("**用户画像（长期记忆）**")
        st.json(state.get("user_profile") or {})


def main() -> None:
    st.set_page_config(page_title="家庭出行管家 · 多智能体", page_icon="🧳", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero">'
        "<h1>🧳 家庭出行管家</h1>"
        "<p>Supervisor–Worker 多智能体 · Router → Planner → 并发 Workers → "
        "Assembler → Critic ⇄ Replan</p>"
        '<div class="badges">'
        '<span class="badge">GraphRAG 多跳检索</span>'
        '<span class="badge">Text2Cypher 防御性纠错</span>'
        '<span class="badge">确定性物化 · 零幻觉点位</span>'
        '<span class="badge">硬约束自愈回跳</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ⚙️ 运行配置")
        mode = st.radio(
            "运行模式",
            ["⚡ 快速模式（确定性）", "🤖 智能体增强（LLM）"],
            index=0,
            help="快速模式秒级、无需 API Key；增强模式调用 LLM，较慢但更智能。",
        )
        enable_llm_critic = st.checkbox("启用 LLM 语义复核（Critic）", value=False)
        st.markdown("### 🔌 当前后端")
        st.markdown(backend_status())
        st.markdown(f"模型：`{DEFAULT_MODEL}`")
        has_key = bool(get_api_key())
        if mode.startswith("🤖") and not has_key:
            st.error("未检测到 API Key，请配置 apik.txt 或 OPENROUTER_API_KEY")
        with st.expander("架构说明"):
            st.markdown(
                "1. **Router**：意图识别 + 硬约束抽取（规则基线 + LLM 增强）\n"
                "2. **Planner**：只生成骨架，规避幻觉\n"
                "3. **Workers**：GraphRAG / 向量 / 天气 **三路并发**（asyncio.gather）\n"
                "4. **Assembler**：确定性物化，杜绝幻觉点位\n"
                "5. **Critic**：规则校验（零容错）+ 可选 LLM 语义复核\n"
                "6. **Replan**：拦截 → 触发 → 局部 State 更新 → 回跳"
            )

    st.markdown('<div class="section-title">① 描述出行需求</div>', unsafe_allow_html=True)
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
        submitted = col_btn.form_submit_button(
            "🚀 生成行程", type="primary", use_container_width=True
        )
        col_hint.caption("快速模式 1–3 秒；智能体增强约 1–9 分钟（需 API Key）")

    if submitted:
        q = (st.session_state.get("query_text") or "").strip()
        if not q:
            st.warning("请输入出行需求")
        else:
            os.environ["FAST_MODE"] = "0" if mode.startswith("🤖") else "1"
            os.environ["ENABLE_LLM_CRITIC"] = "1" if enable_llm_critic else "0"
            started = time.time()
            with st.status("多智能体编排中…", expanded=True) as status:
                st.write("Router → Planner → Workers → Assembler → Critic ⇄ Replan")
                st.caption("快速模式通常 1–3 秒；增强模式约 1–9 分钟。")
                try:
                    trip_state = run_trip(q, session_id="web")
                except Exception as exc:  # noqa: BLE001
                    status.update(label="编排失败", state="error", expanded=True)
                    st.error(f"运行失败：{type(exc).__name__}: {exc}")
                    st.stop()
                elapsed = time.time() - started
                status.update(label=f"编排完成 · {elapsed:.1f}s", state="complete", expanded=False)
            st.session_state["last_state"] = trip_state
            st.session_state["last_elapsed"] = elapsed
            st.session_state["last_query"] = q
            st.toast("行程已生成", icon="✅")

    state = st.session_state.get("last_state")
    if not state:
        st.markdown(
            '<div class="empty">输入需求或点选一个示例，然后点击「生成行程」开始。</div>',
            unsafe_allow_html=True,
        )
        return

    elapsed = float(st.session_state.get("last_elapsed") or 0.0)
    render_kpis(state, elapsed)

    tab_plan, tab_retrieval, tab_check, tab_arch = st.tabs(
        ["📋 行程", "🔎 检索依据", "🛡️ 校验与回跳", "🕸️ 架构与记忆"]
    )

    with tab_plan:
        col_dl, _ = st.columns([1, 3])
        col_dl.download_button(
            "⬇️ 导出 Markdown",
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
