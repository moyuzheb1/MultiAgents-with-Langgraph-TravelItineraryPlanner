from langchain_core.messages import HumanMessage
import json

try:
    from .llm_config import get_llm
except ImportError:
    from llm_config import get_llm

def recommend_activities(state):
    llm = get_llm()
    prompt = f"""
    请根据以下旅行偏好和行程，用中文推荐当地有特色的活动：
    偏好：{json.dumps(state['preferences'], indent=2, ensure_ascii=False)}
    行程：{state['itinerary']}

    要求：
    1. 尽量按天给出建议；
    2. 使用 Markdown 无序列表；
    3. 重点推荐小众、有本地特色的体验。
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"activity_suggestions": result.strip()}
    except Exception as e:
        return {"activity_suggestions": "", "warning": str(e)}