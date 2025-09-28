import sqlite3
import pytest
import sqlite_vec
from sentence_transformers import SentenceTransformer

from rag_experiment.context_based_answering_agent import ContextBasedAnsweringAgent
from rag_experiment.ingestion import chunk_docs, create_schema, ingest_docs, load_docs
from rag_experiment.query_context_provider import QueryContextProvider
from rag_experiment.root_agent import RootAgent
from rag_experiment.text_relevance_checker import TextRelevanceChecker


@pytest.fixture(scope="module")
def agent() -> RootAgent:
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    create_schema(conn)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    docs = load_docs("docs")
    chunks = chunk_docs(docs, max_tokens=500, overlap_tokens=100)
    ingest_docs(conn, chunks, encoder)

    context_based_answering_agent = ContextBasedAnsweringAgent()
    text_relevance_checker = TextRelevanceChecker()
    query_context_provider = QueryContextProvider(conn, encoder, text_relevance_checker)
    return RootAgent(
        context_based_answering_agent=context_based_answering_agent,
        query_context_provider=query_context_provider)

def test_food(agent: RootAgent):
    response = agent.respond("I need to cook some food. What should I do?")
    print_response(response)
    assert response.found_answer is True

def test_programming_languages(agent: RootAgent):
    response = agent.respond("what are some general-purpose programming languages?")
    print_response(response)
    assert response.found_answer is True

def test_python_and_java(agent: RootAgent):
    response = agent.respond("compare python and java")
    print_response(response)
    assert response.found_answer is False

def test_better_programmer(agent: RootAgent):
    response = agent.respond("how do I become a better programmer?")
    print_response(response)
    assert response.found_answer is False

def test_rebel1(agent: RootAgent):
    response = agent.respond("how often should I change the oil in my rebel 500?")
    print_response(response)
    assert response.found_answer is True

def test_rebel2(agent: RootAgent):
    response = agent.respond("anything I should keep in mind when changing motor oil?")
    print_response(response)
    assert response.found_answer is True

def test_rebel3(agent: RootAgent):
    response = agent.respond("give me an example of a motorcycle")
    print_response(response)
    assert response.found_answer is True

def print_response(response):
    print(f"Found Answer: {response.found_answer}")
    print(f"Answer: {response.answer}")
