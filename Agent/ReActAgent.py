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

            # 2. 把 assistant 消息（含 tool_calls）追加回对话历史
            assistant_msg = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # 3. 没有 tool_calls 说明模型已给出最终答案
            if not message.tool_calls:
                final_answer = message.content
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            # 4. 依次执行每个工具调用，并把结果作为 tool 消息追加
            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments or "{}")
                print(f"🎬 行动: {tool_name}[{tool_args}]")

                tool_function = self.tool_executor.getTool(tool_name)
                if not tool_function:
                    observation = f"错误:未找到名为 '{tool_name}' 的工具。"
                else:
                    observation = str(tool_function(**tool_args))  # 调用真实工具

                print(f"👀 观察: {observation}")

                # 记录到历史，并作为 tool 消息返回给模型
                self.history.append(f"Action: {tool_name}[{tool_args}]")
                self.history.append(f"Observation: {observation}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

        print("已达到最大步数，流程终止。")
        return None


if __name__ == "__main__":
    try:
        # 初始化LLM客户端
        llm = HelloAgentsLLM()
        # 初始化工具执行器
        toolExecutor = ToolExecutor()
        # 注册我们的实战搜索工具
        search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
        search_parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要搜索的关键词或问题"},
            },
            "required": ["query"],
        }
        toolExecutor.registerTool("Search", search_description, search, parameters=search_parameters)
        # 初始化ReAct智能体
        agent = ReActAgent(llm_client=llm, tool_executor=toolExecutor)
        # 运行智能体（run 内部已打印最终答案）
        agent.run("英伟达最新的GPU型号是什么")
    except ValueError as e:
        print(e)
