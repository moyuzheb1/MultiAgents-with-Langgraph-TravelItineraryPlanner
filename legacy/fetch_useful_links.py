import os

from dotenv import load_dotenv

load_dotenv()


def _get_search():
    """没有配置 SERPER_API_KEY 时返回 None，避免整个应用被阻塞。"""
    if not os.getenv("SERPER_API_KEY", "").strip():
        return None
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper

        return GoogleSerperAPIWrapper()
    except Exception:
        return None


def fetch_useful_links(state):
    search = _get_search()
    if search is None:
        return {
            "useful_links": [],
            "warning": "未配置 SERPER_API_KEY，已跳过「实用链接」。可在 .env 中填入 Serper Key 后启用。",
        }

    destination = state['preferences'].get('destination', '')
    month = state['preferences'].get('month', '')
    query = f"{destination} {month} 旅行攻略 旅游指南"
    try:
        search_results = search.results(query)
        organic_results = search_results.get("organic", [])
        links = [
            {"title": result.get("title", "无标题"), "link": result.get("link", "")}
            for result in organic_results[:5]
        ]
        return {"useful_links": links}
    except Exception as e:
        return {"useful_links": [], "warning": f"获取链接失败：{str(e)}"}