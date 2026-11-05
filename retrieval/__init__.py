"""可插拔检索层。

三个后端均为「真实服务优先、进程内回退」：
    Neo4j  ← NEO4J_URI
    Milvus ← MILVUS_URI
    Redis  ← REDIS_URL
未配置或连不通时自动回退到零依赖的进程内实现，保证系统开箱可运行。
"""
from retrieval.cache import get_cache
from retrieval.graph_store import CypherError, get_graph_store, parse_cypher
from retrieval.text2cypher import RetrievalResult, retrieve, retrieve_sync
from retrieval.vector_store import get_vector_store

__all__ = [
    "get_cache",
    "get_graph_store",
    "get_vector_store",
    "parse_cypher",
    "CypherError",
    "retrieve",
    "retrieve_sync",
    "RetrievalResult",
]
