"""Supervisor-Worker 多智能体 DAG。

router → planner → workers(asyncio.gather) → assembler → critic ⇄ replan → finalize
"""
from agents.supervisor import build_graph, get_graph, run_trip

__all__ = ["build_graph", "get_graph", "run_trip"]
