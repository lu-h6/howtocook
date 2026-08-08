from typing import Dict, Any

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable, parameters: dict = None):
        """
        向工具箱中注册一个新工具。
        parameters: 可选，OpenAI function calling 所需的参数 JSON Schema。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func, "parameters": parameters}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

    def getToolSchemas(self) -> list:
        """
        获取所有工具的 OpenAI 原生 function calling 工具 Schema 列表。
        """
        schemas = []
        for name, info in self.tools.items():
            parameters = info.get("parameters") or {"type": "object", "properties": {}}
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": parameters,
                },
            })
        return schemas
