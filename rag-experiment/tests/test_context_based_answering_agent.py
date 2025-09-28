import pytest
from rag_experiment.context_based_answering_agent import ContextBasedAnsweringAgent

@pytest.fixture
def agent():
    return ContextBasedAnsweringAgent()

def test_positive(agent: ContextBasedAnsweringAgent):
    assert agent.respond("What is the capital of France?", [
        "Paris is the capital and most populous city of France.",
        "Tuna sandwiches are delicious."
    ]).found_answer is True

def test_negative(agent: ContextBasedAnsweringAgent):
    assert agent.respond("What is the capital of Thailand?", [
        "Paris is the capital and most populous city of France.",
        "Tuna sandwiches are delicious."
    ]).found_answer is False
