from contextlib import closing
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List
from dataclasses import dataclass

from rag_experiment.text_relevance_checker import TextRelevanceChecker


@dataclass
class QueryContextProvider:
    conn: sqlite3.Connection
    encoder: SentenceTransformer
    text_relevance_checker: TextRelevanceChecker

    def get_context(self, query_text: str) -> List[str]:
        query_embedding = self.encoder.encode([query_text])[0]
        query_embedding_blob = np.asarray(query_embedding, dtype=np.float32).tobytes()
        with closing(self.conn.cursor()) as cur:
            cur.execute("""
                SELECT
                    d.id,
                    d.content,
                    e.distance
                FROM embeddings e
                JOIN chunks d ON e.doc_id = d.id
                WHERE e.embedding MATCH ? AND k = 10
                ORDER BY e.distance
            """, (sqlite3.Binary(query_embedding_blob),))

            vector_results = cur.fetchall()

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
            relevance_check_result = self.text_relevance_checker.check(query_text, result['content'])
            print(f"  ID: {result['doc_id']}, Content: '{result['content'][:100]}...'".replace("\n", " "))
            print(f"  Distance: {result['distance']:.4f}, Cross-Encoder Score: {result['cross_score']:.4f}, " +
                  f"Relevant: {relevance_check_result.is_relevant}, Reason: {relevance_check_result.reason}")
            print()

            if relevance_check_result.is_relevant:
                context.append(result['content'])

        return context
