import pytest
from pydantic import ValidationError

from agent_toolkit.schemas import FinalAnswer, final_answer_tool_schema


class TestFinalAnswer:
    def test_valid_answer_parses(self):
        answer = FinalAnswer(
            answer="The result is 42.",
            confidence=0.95,
            citations=[{"source_id": "calculator", "note": "computed the value"}],
        )
        assert answer.confidence == 0.95
        assert answer.citations[0].source_id == "calculator"

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            FinalAnswer(answer="x", confidence=1.5, citations=[])

    def test_missing_answer_rejected(self):
        with pytest.raises(ValidationError):
            FinalAnswer(confidence=0.5, citations=[])

    def test_citations_default_to_empty_list(self):
        answer = FinalAnswer(answer="x", confidence=0.5)
        assert answer.citations == []


class TestFinalAnswerToolSchema:
    def test_schema_has_expected_shape(self):
        schema = final_answer_tool_schema()
        assert schema["name"] == "final_answer"
        assert "answer" in schema["input_schema"]["properties"]
        assert "confidence" in schema["input_schema"]["properties"]
