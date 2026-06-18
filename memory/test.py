# 配置好同级文件夹下.env中的大模型API
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool


class MoonshotCompatibleLLM(HelloAgentsLLM):
    """兼容 Moonshot 对空 assistant 消息的校验。"""

    def invoke(self, messages, **kwargs):
        messages = [
            message
            for message in messages
            if not (
                message.get("role") == "assistant"
                and not (message.get("content") or "").strip()
            )
        ]
        return super().invoke(messages, **kwargs)


# 创建LLM实例
# 注意：必须显式指定 provider="kimi"，否则环境变量里的 DASHSCOPE_API_KEY 会让框架误判为 qwen，
# 把阿里云的 key 发到 Moonshot 接口导致 401；kimi-k2.6 是推理模型，仅支持 temperature=1。
llm = MoonshotCompatibleLLM(provider="kimi", temperature=1)

# 创建工具注册表
tool_registry = ToolRegistry()

# 添加记忆工具
memory_tool = MemoryTool(user_id="user123")
tool_registry.register_tool(memory_tool)

# 添加RAG工具
rag_tool = RAGTool(knowledge_base_path="./knowledge_base")
tool_registry.register_tool(rag_tool)

# 创建Agent并配置工具
agent = SimpleAgent(
    name="智能助手",
    llm=llm,
    system_prompt="你是一个有记忆和知识检索能力的AI助手",
    tool_registry=tool_registry
)

# 开始对话
response = agent.run("你好！请记住我叫张三，我是一名Python开发者")
print(response)
