# HowtoCook Agent

基于 LangChain 与 ReAct 的菜谱问答智能体。

## 结构

- Agent/ 智能体实现：工具定义、工具执行器、ReAct 循环、示例客服
- RAG/ 菜谱知识库的数据清洗与切分
- memory/ 会话记忆
- 	est/ 单元测试

## 运行

`ash
uv sync
uv run main.py
`

需要在 .env 中配置模型与检索服务的密钥（参考 .env.example）。
