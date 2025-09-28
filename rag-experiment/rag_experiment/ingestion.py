from contextlib import closing
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List
from dataclasses import dataclass

from rag_experiment.chunker import chunk_text


@dataclass
class SourceDocument:
    filename: str
    content: str


@dataclass
class DocumentChunk:
    source_file: str
    content: str


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


def chunk_docs(docs: List[SourceDocument],
               max_tokens: int,
               overlap_tokens: int) -> List[DocumentChunk]:

    doc_chunks = []
    for doc in docs:
        normalized_content = '\n'.join(line.strip() for line in doc.content.split('\n') if line.strip())
        if not normalized_content:
            continue

        chunks = chunk_text(normalized_content, max_tokens, overlap_tokens)
        for chunk in chunks:
            doc_chunk = DocumentChunk(
                source_file=doc.filename,
                content=chunk
            )
            doc_chunks.append(doc_chunk)

    return doc_chunks

def create_schema(conn: sqlite3.Connection):
    with closing(conn.cursor()) as cur:
        cur.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL
        )
        """)

        cur.execute("""
        CREATE VIRTUAL TABLE embeddings USING vec0(
            doc_id INTEGER,
            embedding FLOAT[384]
        )
        """)
        conn.commit()

def ingest_docs(conn: sqlite3.Connection,
                chunks: List[DocumentChunk],
                encoder: SentenceTransformer):

    chunk_contents = [chunk.content for chunk in chunks]
    embeddings = encoder.encode(chunk_contents)

    with closing(conn.cursor()) as cur:
        for id, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks (id, content, source) VALUES (?, ?, ?)",
                (id, chunk.content, chunk.source_file)
            )

            blob = np.asarray(emb, dtype=np.float32).tobytes()
            cur.execute(
                "INSERT INTO embeddings (doc_id, embedding) VALUES (?, ?)",
                (id, sqlite3.Binary(blob))
            )
        conn.commit()
