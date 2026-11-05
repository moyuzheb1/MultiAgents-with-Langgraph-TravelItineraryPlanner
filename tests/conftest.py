import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 测试统一走确定性快速模式，不依赖网络与 API Key
os.environ.setdefault("FAST_MODE", "1")
os.environ.setdefault("ENABLE_LLM_CRITIC", "0")
