from langchain_core.messages import HumanMessage

try:
    from .llm_config import get_llm
except ImportError:
    from llm_config import get_llm

def packing_list_generator(state):
    llm = get_llm()
    prompt = f"""
    请为一次旅行生成详细的中文行李清单：
    目的地：{state['preferences'].get('destination', '')}
    月份：{state['preferences'].get('month', '')}
    天数：{state['preferences'].get('duration', 0)} 天
    旅行类型：{state['preferences'].get('holiday_type', '常规')}

    要求：
    1. 结合当地天气和旅行类型列出必备物品；
    2. 按「证件与财物」「衣物」「洗漱与药品」「电子设备」「其他」分类；
    3. 使用 Markdown 无序列表。
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"packing_list": result.strip()}
    except Exception as e:
        return {"packing_list": "", "warning": str(e)}