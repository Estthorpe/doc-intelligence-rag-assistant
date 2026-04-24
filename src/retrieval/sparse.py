# src/retrieval/sparse.py
"""
Sparse retrieval using BM25 keyword matching.

BM25 is the production standard for keyword search. It scores documents
based on term frequency (how often a term appears in a document) and
inverse document frequency (how rare the term is across all documents).

Hybrid retrieval (dense + sparse) is the production standard precisely
because neither alone is sufficient for legal/enterprise text.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import psycopg2

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.retrieval.dense import RetrievedChunk

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace and punctuation tokeniser.
    Lowercases and removes punctuation for BM25 matching.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25:
    """
    BM25 implementation for in-memory corpus scoring.

    BM25 parameters:
        k1=1.5: term frequency saturation. Higher = more weight on
                repeated terms. 1.5 is the standard value.
        b=0.75:  document length normalisation. 0.75 is standard.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0.0

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """
        Fit BM25 on a corpus of (doc_id, text) pairs.

        Args:
            documents: List of (chunk_id, text) tuples.
        """
        self.corpus = []
        self.doc_ids = []

        for doc_id, text in documents:
            tokens = tokenize(text)
            self.corpus.append(tokens)
            self.doc_ids.append(doc_id)

        total_length = sum(len(doc) for doc in self.corpus)
        self.avg_dl = total_length / len(self.corpus) if self.corpus else 1.0

        # Document frequency: how many documents contain each term
        self.df = {}
        for doc_tokens in self.corpus:
            for term in set(doc_tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # IDF: inverse document frequency with BM25 smoothing
        n = len(self.corpus)
        self.idf = {term: math.log((n - df + 0.5) / (df + 0.5) + 1) for term, df in self.df.items()}

    def score(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Score all documents against the query, return top-k.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, bm25_score) sorted by score descending.
        """
        query_terms = tokenize(query)
        scores: list[tuple[str, float]] = []

        for i, doc_tokens in enumerate(self.corpus):
            doc_len = len(doc_tokens)
            term_freq = Counter(doc_tokens)
            score = 0.0

            for term in query_terms:
                if term not in self.idf:
                    continue
                tf = term_freq.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
                score += self.idf[term] * (numerator / denominator)

            if score > 0:
                scores.append((self.doc_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def sparse_search(
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    BM25 keyword search over all chunks in pgvector.

    Loads all chunks from the database, scores them with BM25,
    returns top-k as RetrievedChunk objects.

    Note: This loads all chunks into memory. At portfolio scale
    (50-5000 chunks) this is fine. At production scale (millions
    of chunks) you would use Elasticsearch or OpenSearch instead.

    Args:
        query:  The user's question.
        top_k:  Number of results. Defaults to settings value.

    Returns:
        List of RetrievedChunk objects sorted by BM25 score descending.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k

    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, content, source_doc_id, source_path,
                       chunk_index, token_count, metadata
                FROM document_chunks
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        logger.warning("Sparse search: no chunks in database")
        return []

    chunk_map: dict[str, tuple[str, str, str, int, int, dict[str, object]]] = {}
    documents: list[tuple[str, str]] = []

    for row in rows:
        chunk_id, content, source_doc_id, source_path, chunk_index, token_count, metadata_raw = row
        chunk_id_str = str(chunk_id)
        metadata: dict[str, object] = metadata_raw if isinstance(metadata_raw, dict) else {}
        chunk_map[chunk_id_str] = (
            content,
            source_doc_id,
            source_path,
            chunk_index,
            token_count,
            metadata,
        )
        documents.append((chunk_id_str, content))

    bm25 = BM25()
    bm25.fit(documents)
    scored = bm25.score(query, top_k)

    results: list[RetrievedChunk] = []
    for rank, (chunk_id, score) in enumerate(scored, start=1):
        content, source_doc_id, source_path, chunk_index, token_count, metadata = chunk_map[
            chunk_id
        ]
        results.append(
            RetrievedChunk(
                id=chunk_id,
                content=content,
                source_doc_id=source_doc_id,
                source_path=source_path,
                chunk_index=chunk_index,
                token_count=token_count,
                similarity_score=score,
                sparse_rank=rank,
                metadata=metadata,
            )
        )

    logger.info(f"Sparse search: '{query[:50]}...' → {len(results)} results (top_k={top_k})")
    return results
