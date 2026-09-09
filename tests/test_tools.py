import pytest
from pydantic import ValidationError

from agent_toolkit.tools.base import ToolError
from agent_toolkit.tools.calculator import CalculatorArgs, CalculatorTool
from agent_toolkit.tools.knowledge_base import KnowledgeBaseTool


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
        with pytest.raises(ValidationError):
            CalculatorArgs.model_validate({"expression": [1, 2, 3]})


class TestKnowledgeBaseTool:
    def test_finds_relevant_doc(self):
        tool = KnowledgeBaseTool()
        result = tool({"query": "idempotent retryable operation"})
        assert result["results"]
        assert any("idempotency" in r["id"] for r in result["results"])

    def test_no_match_returns_empty_results(self):
        tool = KnowledgeBaseTool()
        result = tool({"query": "xyzzy nonsense query zzz"})
        assert result["results"] == []

    def test_top_k_is_respected(self):
        tool = KnowledgeBaseTool()
        result = tool({"query": "the a is", "top_k": 1})
        assert len(result["results"]) <= 1
