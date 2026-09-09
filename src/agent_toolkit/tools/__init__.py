from agent_toolkit.tools.base import BaseTool, ToolError, ToolRegistry
from agent_toolkit.tools.calculator import CalculatorTool
from agent_toolkit.tools.knowledge_base import KnowledgeBaseTool
from agent_toolkit.tools.weather import WeatherTool


def default_registry() -> ToolRegistry:
    """The standard tool set the agent ships with."""
    return ToolRegistry([CalculatorTool(), KnowledgeBaseTool(), WeatherTool()])


__all__ = [
    "BaseTool",
    "CalculatorTool",
    "KnowledgeBaseTool",
    "ToolError",
    "ToolRegistry",
    "WeatherTool",
    "default_registry",
]
