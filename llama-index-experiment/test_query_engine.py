from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def test_query_engine():
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

    response = query_engine.query("is python a snake?")
    print(f"Response: {response}")

    for i, source in enumerate(response.source_nodes, 1):
        print(f"Source {i}: {source.node.metadata.get('file_name', 'Unknown')} (relevance: {getattr(source, 'score', 'N/A')})")
