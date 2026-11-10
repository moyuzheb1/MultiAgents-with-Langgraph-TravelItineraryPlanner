"""Text2Cypher 防御性纠错链路。

四步 Fallback Pipeline：
    生成 → AST Schema 校验 → LLM Rewrite → 向量降级

每一步的失败都会被记录到 attempts 中，形成可观测链路（供 LangSmith / 评测体系分析瓶颈节点）。
"""
import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field

from core.llm import get_llm, strip_code_fence
from retrieval.graph_store import CypherError, parse_cypher
from retrieval.schema import schema_prompt

# 从模型输出中精准抽取 MATCH 语句（模型常在 Cypher 外加自然语言或代码块）
_CYPHER_RE = re.compile(r"(MATCH\s.+?)(?:;|\Z)", re.I | re.S)


def _extract_cypher(text: str) -> str:
    text = strip_code_fence(text or "").strip()
    m = _CYPHER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(";")
    idx = text.upper().find("MATCH")
    if idx != -1:
        return text[idx:].strip().rstrip(";")
    return text

GENERATE_PROMPT = """你是 Neo4j Cypher 生成专家。请根据下面的自然语言需求生成**一条** Cypher 查询语句。

{schema}

约束上下文：
{constraints}

自然语言需求：{question}

要求：
1. 只输出 Cypher 本身，不要任何解释文字或代码块标记；
2. 只使用上面 Schema 中出现的标签、关系和属性，禁止臆造属性；
3. 必须包含 LIMIT，且不超过 {limit}；
4. 目标节点标签优先使用 {label_hint}。
"""

REWRITE_PROMPT = """下面这条 Cypher 执行失败，请根据错误原因重写。

{schema}

原始需求：{question}

错误 Cypher：
{cypher}

错误信息：
{error}

要求：只输出修正后的 Cypher，不要解释文字或代码块标记。修正要点：
- 删除 Schema 中不存在的属性与标签；
- 距离酒店的条件只能写在 NEAR 关系的 max_km 属性里；若错误涉及 near.max_km 或任何
  关系变量点属性，删除该 WHERE 条件并把距离移回关系，不要在 WHERE 里访问关系变量；
- WHERE 条件只能写成 n.属性 运算符 值，属性必须属于该标签的白名单；
- 确保比较运算两侧类型一致；
- 保留 LIMIT，只输出一条单段 MATCH 语句。
"""


@dataclass
class RetrievalResult:
    rows: list = field(default_factory=list)
    source: str = "none"          # graph / vector
    status: str = "error"         # ok / rewritten / degraded / error
    cypher: str = ""
    attempts: list = field(default_factory=list)
    error: str = ""
    latency_ms: int = 0


def _cache_key(question: str, label_hint: str, constraints: str) -> str:
    raw = f"{label_hint}|{question}|{constraints}"
    return "t2c:" + hashlib.md5(raw.encode("utf-8")).hexdigest()


async def _generate_cypher(question: str, constraints: dict, label_hint: str,
                           limit: int) -> tuple[str, str]:
    """返回 (Cypher, 错误)；异常不再被吞成空串，链路里能看到真实原因。"""
    try:
        llm = get_llm(temperature=0)
        prompt = GENERATE_PROMPT.format(
            schema=schema_prompt(),
            constraints=constraints or "无",
            question=question,
            limit=limit,
            label_hint=label_hint,
        )
        resp = await llm.ainvoke([{"role": "user", "content": prompt}])
        return _extract_cypher(resp.content), ""
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


async def _rewrite_cypher(question: str, cypher: str, error: str) -> tuple[str, str]:
    try:
        llm = get_llm(temperature=0)
        prompt = REWRITE_PROMPT.format(
            schema=schema_prompt(), question=question, cypher=cypher, error=error
        )
        resp = await llm.ainvoke([{"role": "user", "content": prompt}])
        return _extract_cypher(resp.content), ""
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def _vector_downgrade(question: str, label_hint: str, top_k: int, vector_store) -> list[dict]:
    hits = vector_store.search(question, top_k=top_k, label=label_hint)
    return [h.as_dict() for h in hits]


async def retrieve(
    question: str,
    label_hint: str = "POI",
    top_k: int = 10,
    constraints: dict = None,
    graph_store=None,
    vector_store=None,
    cache=None,
    max_rewrite: int = 1,
) -> RetrievalResult:
    """执行四步 Fallback Pipeline，返回检索结果与完整链路轨迹。"""
    started = time.time()
    result = RetrievalResult()
    constraints = constraints or {}

    cache_key = _cache_key(question, label_hint, str(sorted(constraints.items())))
    if cache is not None:
        cached = cache.get(cache_key)
        if cached:
            cached["source"] = cached.get("source", "graph") + "(cache)"
            cached["latency_ms"] = int((time.time() - started) * 1000)
            return RetrievalResult(**cached)

    cypher = ""
    last_error = ""
    for attempt in range(max_rewrite + 1):
        step = "llm_generate" if attempt == 0 else "llm_rewrite"
        if attempt == 0:
            cypher, llm_error = await _generate_cypher(
                question, constraints, label_hint, top_k)
        else:
            cypher, llm_error = await _rewrite_cypher(question, cypher, last_error)
        if llm_error:
            last_error = llm_error
            result.attempts.append({"step": step, "attempt": attempt,
                                    "status": "error", "error": llm_error})
        if not cypher:
            result.attempts.append({"step": step, "attempt": attempt,
                                    "status": "empty", "error": last_error})
            continue
        try:
            # 第 2 步：AST / Schema 校验
            parse_cypher(cypher)
            result.attempts.append({"step": "validate", "attempt": attempt, "status": "ok", "cypher": cypher})

            # 执行
            rows = graph_store.execute(cypher)
            result.attempts.append({"step": "execute", "attempt": attempt, "status": "ok", "rows": len(rows)})

            if not rows:
                # 语法正确但无结果，降级到向量检索补全召回
                last_error = "图谱查询无结果"
                result.attempts.append({"step": "execute", "attempt": attempt, "status": "empty"})
                break

            result.rows = rows
            result.cypher = cypher
            result.source = "graph"
            result.status = "ok" if attempt == 0 else "rewritten"
            result.latency_ms = int((time.time() - started) * 1000)
            if cache is not None:
                cache.set(cache_key, {
                    "rows": result.rows, "source": result.source, "status": result.status,
                    "cypher": result.cypher, "attempts": result.attempts, "error": "",
                })
            return result

        except CypherError as e:
            last_error = str(e)
            result.attempts.append({
                "step": "validate" if "校验" not in str(e) else "validate",
                "attempt": attempt, "status": "error", "cypher": cypher, "error": last_error,
            })
        except Exception as e:  # 执行期异常同样进入 Rewrite / 降级
            last_error = f"{type(e).__name__}: {e}"
            result.attempts.append({
                "step": "execute", "attempt": attempt, "status": "error",
                "cypher": cypher, "error": last_error,
            })

    # 第 4 步：向量降级
    try:
        result.rows = _vector_downgrade(question, label_hint, top_k, vector_store)
        result.source = "vector"
        result.status = "degraded"
        result.attempts.append({"step": "vector_downgrade", "status": "ok", "rows": len(result.rows)})
    except Exception as e:
        result.status = "error"
        result.error = f"{last_error} | 向量降级失败: {e}"
        result.attempts.append({"step": "vector_downgrade", "status": "error", "error": str(e)})

    if not result.error:
        result.error = last_error
    result.latency_ms = int((time.time() - started) * 1000)
    return result


def retrieve_sync(*args, **kwargs) -> RetrievalResult:
    """同步包装，供 Streamlit 等同步上下文调用。"""
    return asyncio.run(retrieve(*args, **kwargs))
