"""三层 Memory：会话层 / 行程层 / 用户层，以及面向 Agent 的 Hook 注入机制。

- 会话层（session）：最近若干轮交互，支撑多轮追问与指代消解
- 行程层（itinerary）：当前行程状态与变更日志，支撑行中重规划
- 用户层（user）：长期画像（老人行动力、儿童年龄、饮食禁忌），跨会话复用

每个 Agent 通过 inject(scope) 只拿到自己需要的那部分上下文，避免全量历史灌入造成的意图漂移。
"""
from dataclasses import dataclass, field


@dataclass
class Turn:
    query: str
    intent: str = ""
    answer: str = ""
    summary: str = ""


@dataclass
class SessionMemory:
    """会话层：短期对话记忆。"""
    turns: list = field(default_factory=list)
    max_turns: int = 12

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def recent(self, n: int = 3) -> list[Turn]:
        return self.turns[-n:]


@dataclass
class ItineraryMemory:
    """行程层：当前行程状态与变更日志。"""
    plan: dict = field(default_factory=dict)
    confirmed: list = field(default_factory=list)
    changelog: list = field(default_factory=list)

    def update(self, plan: dict, reason: str = "") -> None:
        self.plan = plan
        if reason:
            self.changelog.append({"reason": reason, "slots": len(plan.get("slots", []))})
            self.changelog = self.changelog[-20:]


@dataclass
class UserProfile:
    """用户层：长期画像。"""
    elderly_count: int = 0
    kid_ages: list = field(default_factory=list)
    mobility: str = ""            # 独立行走 / 需拐杖 / 轮椅
    dietary: list = field(default_factory=list)
    budget_level: str = "中档"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "elderly_count": self.elderly_count,
            "kid_ages": self.kid_ages,
            "mobility": self.mobility,
            "dietary": self.dietary,
            "budget_level": self.budget_level,
            "notes": self.notes,
        }


_HUB: "MemoryHub | None" = None


def get_memory_hub() -> "MemoryHub":
    """全局 MemoryHub 单例（Agent 通过它读写三层记忆）。"""
    global _HUB
    if _HUB is None:
        from retrieval.cache import get_cache
        _HUB = MemoryHub(get_cache())
    return _HUB


class MemoryHub:
    """三层 Memory 的统一入口。"""

    def __init__(self, cache=None):
        self._cache = cache
        self._sessions: dict[str, SessionMemory] = {}
        self._itineraries: dict[str, ItineraryMemory] = {}
        self._users: dict[str, UserProfile] = {}

    # ---- 各层访问器 ----
    def session(self, sid: str) -> SessionMemory:
        return self._sessions.setdefault(sid, SessionMemory())

    def itinerary(self, sid: str) -> ItineraryMemory:
        return self._itineraries.setdefault(sid, ItineraryMemory())

    def user(self, uid: str) -> UserProfile:
        return self._users.setdefault(uid, UserProfile())

    # ---- 更新 ----
    def record_turn(self, sid: str, turn: Turn) -> None:
        self.session(sid).add(turn)
        if self._cache is not None:
            self._cache.set(f"mem:session:{sid}",
                            [t.__dict__ for t in self.session(sid).turns], ttl=3600)

    def update_itinerary(self, sid: str, plan: dict, reason: str = "") -> None:
        self.itinerary(sid).update(plan, reason)
        if self._cache is not None:
            self._cache.set(f"mem:itinerary:{sid}", self.itinerary(sid).plan, ttl=86400)

    def update_user(self, uid: str, **kwargs) -> None:
        profile = self.user(uid)
        for k, v in kwargs.items():
            if hasattr(profile, k) and v not in (None, "", []):
                setattr(profile, k, v)
        if self._cache is not None:
            self._cache.set(f"mem:user:{uid}", profile.to_dict(), ttl=604800)

    # ---- Hook 注入 ----
    def inject(self, sid: str, scope: str) -> str:
        """按 Agent 职责注入对应层的记忆，实现上下文精准注入。"""
        blocks = []
        profile = self._users.get(sid)
        itin = self._itineraries.get(sid)
        sess = self._sessions.get(sid)

        if scope in ("router", "planner", "critic") and profile:
            blocks.append(
                "【用户画像-长期】老人 {eld} 位；儿童年龄 {kid}；行动能力：{mob}；"
                "饮食要求：{diet}；预算：{budget}{notes}".format(
                    eld=profile.elderly_count or "未说明",
                    kid=profile.kid_ages or "未说明",
                    mob=profile.mobility or "未说明",
                    diet="、".join(profile.dietary) or "无",
                    budget=profile.budget_level,
                    notes="；备注：" + "；".join(profile.notes) if profile.notes else "",
                )
            )

        if scope in ("planner", "critic") and itin and itin.plan:
            blocks.append(f"【当前行程-已有 {len(itin.plan.get('slots', []))} 个安排】"
                          + ("；变更记录：" + "、".join(c["reason"] for c in itin.changelog[-3:])
                             if itin.changelog else ""))

        if scope in ("router", "planner") and sess and sess.turns:
            recent = sess.recent(3)
            blocks.append("【近期对话】" + " | ".join(
                f"Q:{t.query}→{t.summary or t.intent}" for t in recent))

        return "\n".join(blocks)
