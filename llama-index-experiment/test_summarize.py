from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import pytest

def make_query_engine(docs_reader: SimpleDirectoryReader) -> BaseQueryEngine:
    llm = Ollama(
        model="gpt-oss:20b",
        temperature=0.1,
        request_timeout=120.0
    )
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        request_timeout=120.0
    )
    docs = docs_reader.load_data()
    index = SummaryIndex.from_documents(docs, embed_model=embed_model)
    return index.as_query_engine(response_mode="tree_summarize", llm=llm)

async def run(query_engine: BaseQueryEngine, query: str):
    print("Query:", query)
    response = await query_engine.aquery(query)
    print("Response:", response)
    return response

@pytest.mark.asyncio
async def test_summarize_document():
    query_engine = make_query_engine(SimpleDirectoryReader(input_files=["docs/food_culture.md"]))
    await run(query_engine, "Summarize this document in 3 sentences.")

@pytest.mark.asyncio
async def test_summarize_document_question():
    query_engine = make_query_engine(SimpleDirectoryReader(input_files=["docs/food_culture.md"]))
    await run(query_engine, "Does the document mention cats?")
    await run(query_engine, "Does the document mention pizza?")

@pytest.mark.asyncio
async def test_summarize_multiple_documents():
    query_engine = make_query_engine(SimpleDirectoryReader(
        input_dir="docs",
        required_exts=[".md"],
        recursive=True
    ))
    await run(query_engine, "Summarize these documents in 3 sentences.")

@pytest.mark.asyncio
async def test_summarize_multiple_documents_question():
    query_engine = make_query_engine(SimpleDirectoryReader(
        input_dir="docs",
        required_exts=[".md"],
        recursive=True
    ))
    await run(query_engine, "Does the document mention cats?")
    await run(query_engine, "Does the document mention New Jersey?")
