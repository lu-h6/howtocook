import os
import time
from typing import List, Dict

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# 加载 .env 文件中的环境变量
load_dotenv()


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("MOONSHOT_MODEL")
        apiKey = apiKey or os.getenv("LLM_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL") or os.getenv("MOONSHOT_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 1,
              max_retries: int = 3, retry_delay: float = 2) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        对连接类错误（网络中断、超时、限流、服务端错误）自动重试。
        """
        # 可重试的瞬态错误；参数错误（如 400）不会重试
        retryable_errors = (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
        )

        print(f"🧠 正在调用 {self.model} 模型...")

