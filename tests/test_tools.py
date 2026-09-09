from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from agent_toolkit.tools import default_registry
from agent_toolkit.tools.base import ToolError
from agent_toolkit.tools.calculator import CalculatorArgs, CalculatorTool
from agent_toolkit.tools.knowledge_base import KnowledgeBaseTool
from agent_toolkit.tools.weather import WeatherTool


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


class TestWeatherTool:
    def test_parses_successful_response(self):
        fake_json = {
            "current_condition": [
                {
                    "temp_C": "27",
                    "FeelsLikeC": "29",
                    "weatherDesc": [{"value": "Partly cloudy"}],
                    "humidity": "80",
                }
            ]
        }
        fake_response = MagicMock()
        fake_response.json.return_value = fake_json
        fake_response.raise_for_status.return_value = None

        with patch("httpx.get", return_value=fake_response):
            tool = WeatherTool()
            result = tool({"location": "Kochi"})

        assert result["location"] == "Kochi"
        assert result["temp_c"] == "27"
        assert result["description"] == "Partly cloudy"

    def test_timeout_raises_tool_error(self):
        with patch("httpx.get", side_effect=httpx.TimeoutException("timed out")):
            tool = WeatherTool()
            with pytest.raises(ToolError):
                tool({"location": "Nowhere"})

    def test_bad_status_raises_tool_error(self):
        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=MagicMock(status_code=503)
        )
        with patch("httpx.get", return_value=fake_response):
            tool = WeatherTool()
            with pytest.raises(ToolError):
                tool({"location": "Somewhere"})

    def test_unexpected_payload_raises_tool_error(self):
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"unexpected": "shape"}
        with patch("httpx.get", return_value=fake_response):
            tool = WeatherTool()
            with pytest.raises(ToolError):
                tool({"location": "Somewhere"})


class TestDefaultRegistry:
    def test_default_registry_has_all_three_tools(self):
        registry = default_registry()
        names = {t.name for t in registry}
        assert names == {"calculator", "knowledge_base_search", "get_weather"}

    def test_no_duplicate_tool_names(self):
        registry = default_registry()
        schemas = registry.anthropic_schemas()
        names = [s["name"] for s in schemas]
        assert len(names) == len(set(names))
