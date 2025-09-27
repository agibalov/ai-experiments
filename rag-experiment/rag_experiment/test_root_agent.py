import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer

from rag_experiment.context_based_answering_agent import ContextBasedAnsweringAgent
from rag_experiment.mess import create_schema, ingest_docs, load_docs
from rag_experiment.query_context_provider import QueryContextProvider
from rag_experiment.root_agent import RootAgent
from rag_experiment.text_relevance_checker import TextRelevanceChecker


def make_root_agent() -> RootAgent:
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    create_schema(conn)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    docs = load_docs("docs")
    ingest_docs(conn, docs, max_tokens=500, overlap_tokens=100, encoder=encoder)

    context_based_answering_agent = ContextBasedAnsweringAgent()
    text_relevance_checker = TextRelevanceChecker()
    query_context_provider = QueryContextProvider(conn, encoder, text_relevance_checker)
    return RootAgent(
        context_based_answering_agent=context_based_answering_agent,
        query_context_provider=query_context_provider)

def test_hello():
    agent = make_root_agent()

    for query_text in ["i need to cook some food. what should I do?",
                       "how do I become a better programmer?",
                       "what are some general-purpose programming languages?",
                       "compare python and java"]:

        response = agent.respond(query_text)
        print(f"Found Answer: {response.found_answer}")
        print(f"Answer: {response.answer}")
        print("\n\n")
