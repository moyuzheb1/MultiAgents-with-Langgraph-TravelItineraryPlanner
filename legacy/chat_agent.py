from langchain_core.messages import HumanMessage
import json
import re

try:
    from .llm_config import get_llm
except ImportError:
    from llm_config import get_llm


def _strip_code_fence(text):
    """去掉模型输出外层的 ```json ... ``` 代码块标记。"""
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped


def chat_node(state):
    llm = get_llm()
    prompt = f"""
    背景信息：
    偏好：{json.dumps(state['preferences'], indent=2, ensure_ascii=False)}
    行程：{state['itinerary']}

    用户问题：
    {state['user_question']}

    请用中文、口语化地回答，给出有针对的见解或建议，回答控制在 200 字以内。
    只输出如下 JSON，不要输出其他内容：
    {{ "chat_response": "你的回答" }}
    """
    try:
        result = _strip_code_fence(llm.invoke([HumanMessage(content=prompt)]).content)
        try:
            parsed = json.loads(result)
            response = parsed.get("chat_response", result)
        except json.JSONDecodeError:
            response = result
        chat_entry = {"question": state['user_question'], "response": response}
        chat_history = state.get('chat_history', []) + [chat_entry]
        return {"chat_response": response, "chat_history": chat_history}
    except Exception as e:
        return {"chat_response": "", "warning": str(e)}