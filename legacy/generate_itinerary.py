from langchain_core.messages import HumanMessage
import json

try:
    from .llm_config import get_llm
except ImportError:
    from llm_config import get_llm

def generate_itinerary(state):
    llm = get_llm()
    prompt = f"""
    请根据以下旅行偏好，生成一份详细的旅行行程：
    {json.dumps(state['preferences'], indent=2, ensure_ascii=False)}

    要求：
    1. 按天分节，每天给出上午、下午、晚上的安排；
    2. 每天推荐具体的餐饮选择；
    3. 预留适当的休息时间；
    4. 全部使用中文，采用 Markdown 格式输出。
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"itinerary": result.strip()}
    except Exception as e:
        return {"itinerary": "", "warning": str(e)}