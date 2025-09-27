import sqlite3
import sqlite_vec
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path
from typing import List, Dict, Any
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic_ai import ModelSettings
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic import BaseModel
import json


def is_relevant(question: str, text: str) -> tuple[bool, str]:
    class Answer(BaseModel):
        relevant: bool
        reason: str
    model = OpenAIChatModel(
        model_name='gpt-oss:20b',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        settings=ModelSettings(temperature=0.0)
    )
    agent = Agent(
        model=model,
        output_type=Answer,
        system_prompt="""
        Given a question and a text, determine if text is relevant to a question or not.
        """.strip()
    )
    response = agent.run_sync(json.dumps({
        "question": question,
        "text": text
    }))
    return (response.output.relevant, response.output.reason)


def rephrase_question(question: str, paraphrases: int) -> List[str]:
    class Answer(BaseModel):
        paraphrases: List[str]
    model = OpenAIChatModel(
        model_name='gpt-oss:20b',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        settings=ModelSettings(temperature=0.0)
    )
    agent = Agent(
        model=model,
        output_type=Answer,
        system_prompt="""
        Given a question and a number of paraphrases, generate paraphrased versions of the question.

        Go beyond simple rewording: use broader or closely related terminology when appropriate.
        For example, "singer" could also be expressed as "musician" or "performer".
        Keep the meaning equivalent but allow variation in vocabulary and phrasing.
        """.strip()
    )
    response = agent.run_sync(json.dumps({
        "question": question,
        "paraphrases": paraphrases
    }))
    return response.output.paraphrases


def answer_question(question: str, context: List[str]) -> tuple[str, bool]:
    class Answer(BaseModel):
        answer: str
        found_answer: bool
    model = OpenAIChatModel(
        model_name='gpt-oss:20b',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        settings=ModelSettings(temperature=0.0)
    )
    agent = Agent(
        model=model,
        output_type=Answer,
        system_prompt="""
        Give a question and a context, provide a concise and accurate answer based on the context.

        Prefer answers explicitly stated in the context.
        If there are none, you should infer and draw conclusions, but only from what is in the context.
        If not enough info, say: "The context does not provide the answer."
        """.strip()
    )
    response = agent.run_sync(json.dumps({
        "question": question,
        "context": context
    }))
    return (response.output.answer, response.output.found_answer)


def test_llm():
    assert is_relevant(
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.")[0] is True
    assert is_relevant(
        "How do I make chicken noodle soup?",
        "Paris is the capital and most populous city of France.")[0] is False


def load_markdown_files(docs_dir: str) -> List[Dict[str, str]]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Directory {docs_dir} does not exist.")
        return []

    markdown_files = []
    for file_path in docs_path.glob("*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                markdown_files.append({
                    'filename': file_path.name,
                    'content': content
                })
                print(f"Loaded: {file_path.name} ({len(content)} characters)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return markdown_files


def chunk_text(text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")

    def tiktoken_len(text: str) -> int:
        return len(encoding.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def process_markdown_to_chunks(markdown_files: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    all_chunks = []
    encoding = tiktoken.get_encoding("cl100k_base")

    for doc in markdown_files:
        filename = doc['filename']
        content = doc['content']

        normalized_content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())

        if not normalized_content:
            print(f"Skipping empty document: {filename}")
            continue

        chunks = chunk_text(normalized_content, max_tokens=500, overlap_tokens=100)
        print(f"Created {len(chunks)} chunks from {filename}")

        for i, chunk in enumerate(chunks):
            token_count = len(encoding.encode(chunk))

            chunk_data = {
                'source_file': filename,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'content': chunk,
                'char_count': len(chunk),
                'token_count': token_count
            }
            all_chunks.append(chunk_data)

    return all_chunks


def test_hello():
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        content TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE VIRTUAL TABLE embeddings USING vec0(
        doc_id INTEGER,
        embedding FLOAT[384]
    )
    """)

    print("🔍 Loading markdown files...")
    markdown_files = load_markdown_files("docs")

    if not markdown_files:
        print("No markdown files found in docs/ directory.")
        assert False, "Please add markdown files to docs/ directory for testing"

    print(f"📄 Found {len(markdown_files)} markdown files")

    print("✂️  Chunking documents...")
    chunks = process_markdown_to_chunks(markdown_files)

    if not chunks:
        print("No chunks created. Exiting test.")
        return

    print(f"Created {len(chunks)} chunks total")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_contents = [chunk['content'] for chunk in chunks]
    embs = model.encode(chunk_contents)

    for i, (chunk_data, emb) in enumerate(zip(chunks, embs)):
        cur.execute(
            "INSERT INTO chunks (id, content) VALUES (?, ?)",
            (i, chunk_data['content'])
        )

        blob = np.asarray(emb, dtype=np.float32).tobytes()
        cur.execute(
            "INSERT INTO embeddings (doc_id, embedding) VALUES (?, ?)",
            (i, sqlite3.Binary(blob))
        )
    conn.commit()

    print()

    for query_text in ["i need to cook some food. what should I do?",
                       "how do I become a better programmer?",
                       "what are some general-purpose programming languages?",
                       "compare python and java"]:

        paraphrases = rephrase_question(query_text, 5)
        print(f"Paraphrases for '{query_text}':")
        for para in paraphrases:
            print(f" - {para}")
        print()

        q = model.encode([query_text])[0]
        q_blob = np.asarray(q, dtype=np.float32).tobytes()

        cur.execute("""
            SELECT
                d.id,
                d.content,
                e.distance
            FROM embeddings e
            JOIN chunks d ON e.doc_id = d.id
            WHERE e.embedding MATCH ? AND k = 10
            ORDER BY e.distance
        """, (sqlite3.Binary(q_blob),))

        vector_results = cur.fetchall()

        # cross-encoder reranking
        reranked_results = []
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
        query_doc_pairs = [(query_text, content) for _, content, _ in vector_results]
        cross_scores = cross_encoder.predict(query_doc_pairs)
        for (doc_id, content, distance), cross_score in zip(vector_results, cross_scores):
            result = {
                'doc_id': doc_id,
                'content': content,
                'distance': distance,
                'cross_score': cross_score
            }
            reranked_results.append(result)

        reranked_results.sort(key=lambda result: result['cross_score'], reverse=True)
        reranked_results = reranked_results[:3]

        print("*" * 80)
        print(f"Query: '{query_text}'")
        print("Results:")
        context = []
        for result in reranked_results:
            is_relevant_flag, reason = is_relevant(query_text, result['content'])
            print(f"  ID: {result['doc_id']}, Content: '{result['content'][:100]}...'".replace("\n", " "))
            print(f"  Distance: {result['distance']:.4f}, Cross-Encoder Score: {result['cross_score']:.4f}, " +
                  f"Relevant: {is_relevant_flag}, Reason: {reason}")
            print()

            if is_relevant_flag:
                context.append(result['content'])

        answer, found_answer = answer_question(question=query_text, context=context)
        print(f"Found Answer: {found_answer}")
        print(f"Answer: {answer}")
        print("\n\n")
