import json

from Agent.hello_agent import HelloAgentsLLM
from Agent.tool import search
from Agent.toolExectuor import ToolExecutor


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        使用原生 function calling 运行智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录

        messages = [
            {"role": "system", "content": "你是一个能调用外部工具来回答问题的智能助手。"},
            {"role": "user", "content": question},
        ]
        tools = self.tool_executor.getToolSchemas()

        for step in range(1, self.max_steps + 1):
            print(f"--- 第 {step} 步 ---")

            # 1. 调用LLM，允许其决定是否调用工具
            message = self.llm_client.chat(messages=messages, tools=tools)
            if not message:
                print("错误:LLM未能返回有效响应。")
                break

