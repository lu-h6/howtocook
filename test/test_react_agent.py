"""
ReActAgent 的 pytest 测试。

不依赖真实的 LLM / 网络 / API key，通过 fake LLM 与 fake 工具执行器
对 ReActAgent 的原生 function calling 流程进行测试。

运行方式（在项目根目录下）:
    .venv/Scripts/python.exe -m pytest test/test_react_agent.py -v
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

# 确保项目根目录在 sys.path 中，使 `Agent.*` 包可被导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Agent.ReActAgent import ReActAgent


def make_tool_call(call_id, name, args_json):
    """构造一个假的 tool_call 对象（含 id 与 function.name/arguments）。"""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=args_json),
    )


def make_message(content=None, tool_calls=None):
    """构造一个假的 assistant message 对象（含 content 与 tool_calls）。"""
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


class FakeLLM:
    """按顺序返回预设 message、并记录调用参数的假 LLM 客户端。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools})
        return self.responses.pop(0) if self.responses else None


class FakeToolExecutor:
    """可注册假工具、并记录调用日志的假工具执行器。"""

    def __init__(self, tools=None):
        # tools: {name: (description, func)}
        self.tools = dict(tools or {})

    def getToolSchemas(self):
        return []  # 测试中不需要真实 Schema

    def getTool(self, name):
        return self.tools[name][1] if name in self.tools else None


def run_quietly(agent, question):
    """静默运行 agent.run，屏蔽过程中的 print 输出。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = agent.run(question)
    return result, buf.getvalue()


def test_run_final_answer_directly():
    llm = FakeLLM([make_message(content="42")])
    executor = FakeToolExecutor()
    agent = ReActAgent(llm_client=llm, tool_executor=executor)

    result, _ = run_quietly(agent, "1+1 等于多少")
    assert result == "42"
    assert len(llm.calls) == 1


def test_run_with_tool_call():
    def fake_search(query):
        return "最新型号是 RTX 5090"

    llm = FakeLLM([
        make_message(
            content=None,
            tool_calls=[make_tool_call("call_1", "Search", '{"query": "英伟达最新GPU"}')],
        ),
        make_message(content="RTX 5090"),
    ])
    executor = FakeToolExecutor({"Search": ("一个网页搜索引擎", fake_search)})
    agent = ReActAgent(llm_client=llm, tool_executor=executor)

    result, _ = run_quietly(agent, "英伟达最新GPU型号")
    assert result == "RTX 5090"
    assert len(llm.calls) == 2

    # 第二次调用时，messages 里应包含 assistant 的 tool_calls 与 tool 结果消息
    second_messages = llm.calls[1]["messages"]
    tool_msg = next(m for m in second_messages if m["role"] == "tool")
    assert tool_msg["content"] == "最新型号是 RTX 5090"
    assert tool_msg["tool_call_id"] == "call_1"


def test_run_tool_not_found():
    llm = FakeLLM([
        make_message(
            content=None,
            tool_calls=[make_tool_call("call_1", "NoSuchTool", '{"x": "y"}')],
        ),
        make_message(content="结果"),
    ])
    executor = FakeToolExecutor()
    agent = ReActAgent(llm_client=llm, tool_executor=executor)

    result, _ = run_quietly(agent, "问题")
    assert result == "结果"
    assert "错误:未找到名为 'NoSuchTool' 的工具。" in agent.history[1]


def test_run_llm_none_response_breaks():
    llm = FakeLLM([None])
    executor = FakeToolExecutor()
    agent = ReActAgent(llm_client=llm, tool_executor=executor)

    result, _ = run_quietly(agent, "问题")
    assert result is None
    assert len(llm.calls) == 1


def test_run_max_steps_reached():
    # LLM 一直调用工具、从不给出最终答案，直到达到最大步数
    llm = FakeLLM([
        make_message(
            content=None,
            tool_calls=[make_tool_call(f"call_{i}", "Search", '{"query": "x"}')],
        )
        for i in range(10)
    ])
    executor = FakeToolExecutor({"Search": ("一个搜索工具", lambda query: "结果")})
    agent = ReActAgent(llm_client=llm, tool_executor=executor, max_steps=3)

    result, _ = run_quietly(agent, "问题")
    assert result is None
    assert len(llm.calls) == 3
