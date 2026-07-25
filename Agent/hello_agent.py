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

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )

                # 处理流式响应
                print("✅ 大语言模型响应成功:")
                collected_content = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    collected_content.append(content)
                print()  # 在流式输出结束后换行
                return "".join(collected_content)

            except retryable_errors as e:
                if attempt < max_retries:
                    print(f"\n⚠️ 连接类错误，{retry_delay} 秒后重试 "
                          f"({attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"\n❌ 重试 {max_retries} 次后仍失败: {e}")
                    return None

            except Exception as e:
                # 非瞬态错误（如 400 参数错误）直接失败，不重试
                print(f"❌ 调用LLM API时发生错误: {e}")
                return None

        return None  # 理论上不会执行到这里

    def chat(self, messages: List[Dict[str, str]], tools: list = None,
             temperature: float = 1, max_retries: int = 3, retry_delay: float = 2):
        """
        调用大语言模型，支持原生工具调用（function calling）。
        返回完整的 assistant message 对象（含 content 与 tool_calls）。
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

        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = tools

                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                print("✅ 大语言模型响应成功")
                return message

            except retryable_errors as e:
                if attempt < max_retries:
                    print(f"\n⚠️ 连接类错误，{retry_delay} 秒后重试 "
                          f"({attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"\n❌ 重试 {max_retries} 次后仍失败: {e}")
                    return None

            except Exception as e:
                # 非瞬态错误（如 400 参数错误）直接失败，不重试
                print(f"❌ 调用LLM API时发生错误: {e}")
                return None

        return None  # 理论上不会执行到这里


# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)

