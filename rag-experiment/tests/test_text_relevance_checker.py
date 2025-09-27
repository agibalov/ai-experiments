import pytest
from rag_experiment.text_relevance_checker import TextRelevanceChecker

@pytest.fixture
def checker():
    return TextRelevanceChecker()

def test_positive(checker: TextRelevanceChecker):
    assert checker.check(
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.").is_relevant is True

def test_negative(checker: TextRelevanceChecker):
    assert checker.check(
        "How do I make chicken noodle soup?",
        "Paris is the capital and most populous city of France.").is_relevant is False
