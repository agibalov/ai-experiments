from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
)
import pytest


@pytest.mark.asyncio
async def test_agent_with_query_engine():
    llm = Ollama(
        model="gpt-oss:20b",
        temperature=0.1,
        request_timeout=120.0
    )
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        request_timeout=120.0
    )

    reader = SimpleDirectoryReader(
        input_dir="docs",
        required_exts=[".md"],
        recursive=True
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode="compact",
        verbose=True
    )

    async def search_documents(query: str) -> str:
        """Useful for searching documents to answer questions."""
        response = await query_engine.aquery(query)
        return str(response)

    llm = Ollama(
        model="gpt-oss:20b",
        temperature=0.1,
        request_timeout=120.0
    )

    agent = FunctionAgent(
        tools=[search_documents],
        llm=llm,
        system_prompt="""You are a helpful assistant that must always use the search_documents tool
        to answer user's questions."""
    )

    handler = agent.run("is python a snake?")

    async for event in handler.stream_events():
        if isinstance(event, ToolCall):
            print(f"ToolCall: {event}")
        elif isinstance(event, ToolCallResult):
            print(f"ToolCallResult: {event}")

    response = await handler

    print("Response:", response)
