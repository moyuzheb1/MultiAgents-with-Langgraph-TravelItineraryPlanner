"""家庭出行管家 · Supervisor-Worker 多智能体系统（Streamlit 可视化控制台）。

功能：
- 两种运行模式：⚡ 快速模式（确定性，秒级）/ 🤖 智能体增强（LLM，较慢）
- 行程结果、检索链路（Text2Cypher Fallback）、Critic 校验报告、Replan 回跳历史
- 三层 Memory 注入可视化、架构 DAG 可视化
"""
from __future__ import annotations

import os

import streamlit as st

from agents import run_trip  # noqa: E402
from core.llm import DEFAULT_MODEL, get_api_key  # noqa: E402
from retrieval import get_cache, get_graph_store, get_vector_store  # noqa: E402

EXAMPLES = [
    "带70岁老人和6岁小孩去深圳玩3天，需要轮椅无障碍，饮食清淡，住深圳福田香格里拉大酒店，避开海鲜排档",
    "深圳2日游，遇到台风，带老人，要避开海鲜排档",
    "一家人去深圳玩4天，小孩3岁，人均餐标100以内",
    "带老人去大梅沙附近玩2天，不要爬山，雨天要有室内备选",
]


def backend_status() -> str:
    g = get_graph_store().backend
    v = get_vector_store().backend
    c = get_cache().backend
    return f"图存储 **{g}** ｜ 向量 **{v}** ｜ 缓存 **{c}**"


def dag_html(executed: set) -> str:
    """用纯 HTML 渲染架构 DAG（避免 graphviz 二进制依赖）。"""
    def box(nid: str, title: str, sub: str) -> str:
        active = "background:#eaf2ff;border-color:#1f77b4;color:#0b3d91" if nid in executed \
            else "background:#f5f5f5;border-color:#bbb;color:#666"
        return (f'<div style="border:2px solid;{active};border-radius:8px;'
                f'padding:8px 10px;text-align:center;min-width:96px;font-size:12px">'
                f'<b>{title}</b><br><span style="font-size:11px">{sub}</span></div>')

    def arrow(label: str = "", color: str = "#333") -> str:
        return (f'<div style="align-self:center;color:{color};font-size:12px;'
                f'margin:0 4px">→<br><span style="font-size:10px">{label}</span></div>')

    chain = "".join([
        box("router", "Router", "意图/约束"), arrow(),
        box("planner", "Planner", "行程骨架"), arrow(),
        box("workers", "并发 Workers", "图谱/向量/天气"), arrow(),
        box("assembler", "Assembler", "物化槽位"), arrow(),
        box("critic", "Critic", "硬约束校验"),
    ])
    loop = (f'{arrow("通过","#2ca02c")}{box("finalize","Finalize","收尾输出")}'
            f'<div style="margin-top:14px;color:#d62728;font-size:12px">'
            f'↺ Critic 检出致命违规 → <b>Replan</b>（拦截→触发→局部 State 更新）'
            f'→ 回跳 Assembler 局部重填 / 或 Planner 全量重规划</div>')
    replan_box = box("replan", "Replan", "局部State更新")
    return (f'<div style="display:flex;flex-wrap:wrap;align-items:center;'
            f'font-family:sans-serif">{chain}{loop}</div>'
            f'<div style="margin-top:18px">{replan_box} '
            f'<span style="color:#d62728;font-size:12px">（条件边，按需插入循环）</span></div>')


def render_trace(trace: list, attempts_by_worker: dict | None = None) -> None:
    if not trace:
        st.info("无检索轨迹")
        return
    rows = []
    for t in trace:
        rows.append({
            "Worker": t.get("worker"),
            "数据源": t.get("source"),
            "状态": t.get("status"),
            "命中数": t.get("rows"),
            "耗时(ms)": t.get("latency_ms"),
            "错误": t.get("error", "")[:60],
        })
    st.table(rows)
    if attempts_by_worker:
        st.markdown("**Text2Cypher Fallback 链路**（生成 → 校验 → 重写 → 向量降级）")
        for name, attempts in attempts_by_worker.items():
            with st.expander(f"{name} 链路细节", expanded=False):
                for a in attempts:
                    st.write(f"- 步骤 `{a.get('step')}` 第{a.get('attempt')}次："
                             f"**{a.get('status')}** {a.get('error','')}")
                    if a.get("rows") is not None:
                        st.caption(f"  执行命中 {a['rows']} 条")


def render_critic(report: dict) -> None:
    if not report:
        st.info("无校验报告")
        return
    v = report.get("violations", [])
    if report.get("passed"):
        st.success(f"✅ 全部硬约束校验通过（致命违规 0 条，提示 {report.get('warning_count',0)} 条）")
    else:
        st.error(f"⛔ 存在 {report.get('fatal_count')} 条致命违规，已触发自动回跳重规划")
    if not v:
        return
    for item in v:
        sev = item.get("severity")
        if sev == "fatal":
            st.error(f"**[{item.get('code')}]** 第{item.get('day')}天 — {item.get('detail')}　→ {item.get('suggestion')}")
        else:
            st.warning(f"**[{item.get('code')}]** 第{item.get('day')}天 — {item.get('detail')}　→ {item.get('suggestion')}")


def render_replan(history: list) -> None:
    if not history:
        st.info("本行程无需重规划（一次通过）")
        return
    for h in history:
        st.markdown(f"**第 {h.get('round')} 轮回跳** → 下一跳 `{h.get('next')}`　"
                    f"｜ 致命违规 {h.get('fatal_count')} 条　违规码 {h.get('codes')}")
        st.caption(f"本次排除实体：{h.get('excluded') or '无'}")


def render_constraints(state: dict) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**抽取硬约束**")
        st.json({k: v for k, v in (state.get("constraints") or {}).items()
                 if not k.startswith("_")})
    with col2:
        st.markdown("**用户画像（长期记忆）**")
        st.json(state.get("user_profile") or {})
    clar = state.get("clarifications") or []
    if clar:
        st.warning("需确认：" + "；".join(clar))


def main() -> None:
    st.set_page_config(page_title="家庭出行管家 · 多智能体", page_icon="🧳", layout="wide")
    st.title("🧳 家庭出行管家 · Supervisor-Worker 多智能体")
    st.caption("Router → Planner → 并发 Workers → Assembler → Critic ⇄ Replan（Text2Cypher 防御性纠错 / 三层 Memory 注入）")

    with st.sidebar:
        st.markdown("### ⚙️ 运行配置")
        mode = st.radio("运行模式", ["⚡ 快速模式（确定性）", "🤖 智能体增强（LLM）"], index=0)
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
                "3. **Workers**：图谱/向量/天气 **三路并发**（asyncio.gather）\n"
                "4. **Assembler**：确定性物化，杜绝幻觉点位\n"
                "5. **Critic**：规则校验（零容错）+ 可选 LLM 语义复核（勾选后生效）\n"
                "6. **Replan**：拦截 → 触发 → 局部 State 更新 → 回跳"
            )

    query = st.text_area("请输入出行需求", height=90,
                         placeholder="例如：带70岁老人和小孩去深圳玩3天，需要轮椅，饮食清淡，住福田香格里拉…")
    col_ex1, col_ex2 = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        c = col_ex1 if i % 2 == 0 else col_ex2
        if c.button(f"示例{i+1}", key=f"ex{i}", use_container_width=True):
            query = ex
            st.session_state["query"] = ex

    run = st.button("🚀 生成行程", type="primary", use_container_width=True)

    if run or st.session_state.get("query"):
        q = st.session_state.get("query", query)
        if not q or not q.strip():
            st.warning("请输入出行需求")
            return
        os.environ["FAST_MODE"] = "0" if mode.startswith("🤖") else "1"
        os.environ["ENABLE_LLM_CRITIC"] = "1" if enable_llm_critic else "0"
        with st.spinner("多智能体编排中…（增强模式较慢，约 1-9 分钟）"):
            state = run_trip(q, session_id="web")
        st.session_state["last_state"] = state
        st.session_state["query"] = ""

    state = st.session_state.get("last_state")
    if not state:
        st.info("在上方输入需求并点击「生成行程」开始。")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📋 行程", "🔎 检索链路", "🛡️ Critic 校验", "🔁 Replan 回跳", "🧠 约束 & 记忆", "🕸️ 架构 DAG"])

    with tab1:
        st.markdown(state.get("answer", ""))

    with tab2:
        trace = state.get("retrieval_trace", [])
        render_trace(trace)
        # 直接展示各 worker 的降级说明
        for name, payload in (state.get("worker_results") or {}).items():
            if payload.get("error"):
                st.caption(f"{name}：{payload['error']}")

    with tab3:
        render_critic(state.get("critic_report", {}))

    with tab4:
        render_replan(state.get("replan_history", []))

    with tab5:
        render_constraints(state)

    with tab6:
        executed = {"router", "planner", "workers", "assembler", "critic", "finalize"}
        if state.get("replan_history"):
            executed.add("replan")
        st.markdown(dag_html(executed), unsafe_allow_html=True)

    # 导出
    st.download_button("⬇️ 导出行程 Markdown",
                       data=state.get("answer", ""),
                       file_name="深圳家庭游行程.md",
                       mime="text/markdown")


if __name__ == "__main__":
    main()
