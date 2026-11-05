from langchain_core.messages import HumanMessage

try:
    from .llm_config import get_llm
except ImportError:
    from llm_config import get_llm

def weather_forecaster(state):
    llm = get_llm()
    prompt = f"""
    请根据目的地和出行月份，用中文提供详细的天气预测，内容需包含：
    1. 气温范围；
    2. 降水情况；
    3. 给旅行者的穿衣与出行建议。

    目的地：{state['preferences'].get('destination', '')}
    月份：{state['preferences'].get('month', '')}

    全部使用中文，采用 Markdown 格式输出。
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"weather_forecast": result.strip()}
    except Exception as e:
        return {"weather_forecast": "", "warning": str(e)}