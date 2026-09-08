import pytest

from agent_toolkit.tools.calculator import CalculatorArgs, CalculatorTool
from agent_toolkit.tools.base import ToolError


class TestCalculatorTool:
    def test_basic_arithmetic(self):
        tool = CalculatorTool()
        result = tool({"expression": "(12 + 4) * 3 / 2"})
        assert result["result"] == 24.0

    def test_rejects_division_by_zero(self):
        tool = CalculatorTool()
        with pytest.raises(ToolError):
            tool({"expression": "1 / 0"})

    def test_rejects_non_arithmetic_input(self):
        tool = CalculatorTool()
        with pytest.raises(ToolError):
            tool({"expression": "__import__('os').system('echo hi')"})

    def test_rejects_missing_argument(self):
        tool = CalculatorTool()
        with pytest.raises(ToolError):
            tool({})

    def test_schema_export_has_expected_shape(self):
        tool = CalculatorTool()
        schema = tool.to_anthropic_schema()
        assert schema["name"] == "calculator"
        assert "expression" in schema["input_schema"]["properties"]

    def test_args_model_validates_types(self):
        with pytest.raises(Exception):
            CalculatorArgs.model_validate({"expression": [1, 2, 3]})
