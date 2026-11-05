"""统一的 LLM 配置：通过 OpenRouter 提供 OpenAI 兼容接口。"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# override=True：防止系统里已存在的同名环境变量（可能是失效的 Key）覆盖 .env
load_dotenv(override=True)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_api_key() -> str:
    """优先读取项目根目录 apik.txt 中的 Key，其次读取环境变量 OPENROUTER_API_KEY。"""
    apik_path = os.path.join(ROOT_DIR, "apik.txt")
    try:
        with open(apik_path, "r", encoding="utf-8") as f:
            key = f.read().strip()
        if key:
            return key
    except OSError:
        pass

    return os.getenv("OPENROUTER_API_KEY", "").strip()


@lru_cache(maxsize=None)
def get_llm(model: str = None, temperature: float = 0.7):
    """返回指向 OpenRouter 的 ChatOpenAI 实例（按模型缓存复用）。"""
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "未找到 OpenRouter API Key。请在项目根目录的 apik.txt 中写入 Key，"
            "或在 .env 中设置 OPENROUTER_API_KEY。"
        )

    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=8192,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AI Travel Itinerary Planner",
        },
    )
