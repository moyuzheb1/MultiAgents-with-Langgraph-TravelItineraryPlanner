from langchain_core.messages import HumanMessage

try:
    from .llm_config import get_llm
except ImportError:
    from llm_config import get_llm

def food_culture_recommender(state):
    llm = get_llm()
    prompt = f"""
    针对一次旅行，请用中文提供美食与文化建议：
    目的地：{state['preferences'].get('destination', '')}
    预算类型：{state['preferences'].get('budget_type', '中等')}

    请包含以下内容：
    1. 当地热门美食与推荐餐厅；
    2. 重要的文化习俗、礼仪注意事项，以及旅行者需要留意的事项。

    使用 Markdown 分节输出，一级标题固定为「美食与餐饮」和「文化与礼仪」。
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"food_culture_info": result.strip()}
    except Exception as e:
        return {"food_culture_info": "", "warning": str(e)}