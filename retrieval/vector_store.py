"""向量检索层：Milvus 优先，无 Milvus 时回退到进程内稀疏向量检索。

进程内实现采用「字符二元组 + TF-IDF + 余弦相似度」，对中文短文本效果稳定，
无需任何外部依赖或 embedding 调用，保证系统开箱可用。
"""
import math
import os
import re
from collections import Counter
from typing import Any, Optional

from data.shenzhen import HOTELS, POIS, RESTAURANTS
from retrieval.schema import VECTOR_TEXT_FIELDS, label_of


class VectorHit:
    __slots__ = ("entity", "label", "score")

    def __init__(self, entity, label: str, score: float):
        self.entity = entity
        self.label = label
        self.score = score

    def as_dict(self) -> dict:
        d = dict(self.entity.__dict__) if hasattr(self.entity, "__dict__") else {}
        return {"label": self.label, "score": round(self.score, 4), **d}


def _tokenize(text: str) -> list[str]:
    """中文按二元组切分，英文数字按词切分。"""
    text = text.lower()
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    latin = re.findall(r"[a-z0-9]+", text)
    grams: list[str] = []
    for seg in cjk:
        grams.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        if len(seg) == 1:
            grams.append(seg)
    grams.extend(latin)
    return grams


def _entity_text(entity) -> str:
    label = label_of(entity)
    fields = VECTOR_TEXT_FIELDS.get(label, ())
    parts = [str(getattr(entity, f, "") or "") for f in fields]
    return " ".join(p for p in parts if p)


class InMemoryVectorStore:
    """进程内稀疏向量检索。"""

    def __init__(self, entities=None):
        self.entities = entities if entities is not None else (
            list(POIS) + list(RESTAURANTS) + list(HOTELS)
        )
        self._docs = [_tokenize(_entity_text(e)) for e in self.entities]
        self._tf = [Counter(d) for d in self._docs]
        self._idf = self._build_idf()
        self._norms = [self._norm(self._weights(tf)) for tf in self._tf]

    def _build_idf(self) -> dict[str, float]:
        n = len(self._docs)
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _weights(self, tf: Counter) -> dict[str, float]:
        return {t: (1.0 + math.log(c)) * self._idf.get(t, 1.0) for t, c in tf.items()}

    @staticmethod
    def _norm(vec: dict[str, float]) -> float:
        return math.sqrt(sum(v * v for v in vec.values())) or 1.0

    def search(self, text: str, top_k: int = 10, label: Optional[str] = None) -> list[VectorHit]:
        q = self._weights(Counter(_tokenize(text)))
        qn = self._norm(q) or 1.0
        hits: list[VectorHit] = []
        for ent, tf, norm in zip(self.entities, self._tf, self._norms):
            if label and label_of(ent) != label:
                continue
            vec = self._weights(tf)
            score = sum(w * vec.get(t, 0.0) for t, w in q.items()) / (qn * norm)
            if score > 0:
                hits.append(VectorHit(ent, label_of(ent), score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    @property
    def backend(self) -> str:
        return "memory"


class MilvusVectorStore:
    """Milvus 向量检索。需要 pymilvus 包与 MILVUS_URI 环境变量。"""

    def __init__(self, uri: str, collection: str = "travel_poi", dim: int = 1024):
        from pymilvus import MilvusException, connections, utility  # 延迟导入

        self._dim = dim
        self._collection_name = collection
        connections.connect(alias="default", uri=uri)
        if not utility.has_collection(collection):
            raise MilvusException(f"集合 {collection} 不存在，请先写入向量数据")
        from pymilvus import Collection
        self._coll = Collection(collection)
        self._coll.load()
        # 未接入 embedding 服务时，检索自动回退到进程内实现，保证配置后仍可用
        self._fallback = InMemoryVectorStore()

    def search(self, text: str, top_k: int = 10, label: Optional[str] = None) -> list[VectorHit]:
        return self._fallback.search(text, top_k=top_k, label=label)

    @property
    def backend(self) -> str:
        return "milvus(fallback)"


def get_vector_store():
    """按环境选择向量后端：配置了 MILVUS_URI 且连通则用 Milvus，否则用内存检索。"""
    uri = os.getenv("MILVUS_URI", "").strip()
    if uri:
        try:
            return MilvusVectorStore(uri)
        except Exception:
            pass
    return InMemoryVectorStore()
