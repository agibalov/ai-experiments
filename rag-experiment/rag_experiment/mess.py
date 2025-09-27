from contextlib import closing
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass



@dataclass
class SourceDocument:
    filename: str
    content: str


@dataclass
class DocumentChunk:
    source_file: str
    chunk_index: int
    total_chunks: int
    content: str
    char_count: int
    token_count: int

def load_docs(docs_dir: str) -> List[SourceDocument]:
    docs_path = Path(docs_dir)
    markdown_files = []
    for file_path in docs_path.glob("*.md"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            source_doc = SourceDocument(
                filename=file_path.name,
                content=content
            )
            markdown_files.append(source_doc)
            print(f"Loaded: {file_path.name} ({len(content)} characters)")

    return markdown_files


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
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


def chunk_docs(docs: List[SourceDocument],
               max_tokens: int,
               overlap_tokens: int) -> List[DocumentChunk]:

    doc_chunks = []
    encoding = tiktoken.get_encoding("cl100k_base")

    for doc in docs:
        normalized_content = '\n'.join(line.strip() for line in doc.content.split('\n') if line.strip())
        if not normalized_content:
            continue

        chunks = chunk_text(normalized_content, max_tokens, overlap_tokens)
        for i, chunk in enumerate(chunks):
            token_count = len(encoding.encode(chunk))

            doc_chunk = DocumentChunk(
                source_file=doc.filename,
                chunk_index=i,
                total_chunks=len(chunks),
                content=chunk,
                char_count=len(chunk),
                token_count=token_count
            )
            doc_chunks.append(doc_chunk)

    return doc_chunks

def create_schema(conn: sqlite3.Connection):
    with closing(conn.cursor()) as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
            doc_id INTEGER,
            embedding FLOAT[384]
        )
        """)
        conn.commit()

def ingest_docs(conn: sqlite3.Connection,
                docs: List[SourceDocument],
                max_tokens: int,
                overlap_tokens: int,
                encoder: SentenceTransformer):

    chunks = chunk_docs(docs, max_tokens, overlap_tokens)
    chunk_contents = [chunk.content for chunk in chunks]
    embeddings = encoder.encode(chunk_contents)

    with closing(conn.cursor()) as cur:
        for id, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks (id, content) VALUES (?, ?)",
                (id, chunk.content)
            )

            blob = np.asarray(emb, dtype=np.float32).tobytes()
            cur.execute(
                "INSERT INTO embeddings (doc_id, embedding) VALUES (?, ?)",
                (id, sqlite3.Binary(blob))
            )
        conn.commit()
