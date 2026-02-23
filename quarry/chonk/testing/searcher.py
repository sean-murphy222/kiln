"""
Retrieval testing engine for CHONK.

The core "test before you embed" feature - allows users to test
queries against their chunks before committing to an embedding service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from chonk.core.document import Chunk, ChonkDocument, ChonkProject, TestQuery, TestSuite
from chonk.testing.embedder import Embedder


@dataclass
class SearchResult:
    """A single search result with chunk and similarity score."""

    chunk: Chunk
    score: float
    rank: int
    document_id: str | None = None
    document_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "score": round(self.score, 4),
            "rank": self.rank,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "content_preview": self.chunk.content[:200],
            "token_count": self.chunk.token_count,
            "page_range": self.chunk.page_range,
            "hierarchy_path": self.chunk.hierarchy_path,
        }


@dataclass
class TestResult:
    """Result of running a test query."""

    query: TestQuery
    results: list[SearchResult]
    passed: bool  # Did expected chunks appear in top-k?
    missing_expected: list[str] = field(default_factory=list)  # Expected but not found
    unexpected_excluded: list[str] = field(default_factory=list)  # Excluded but found
    execution_time_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query.id,
            "query_text": self.query.query,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
            "missing_expected": self.missing_expected,
            "unexpected_excluded": self.unexpected_excluded,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class TestSuiteReport:
    """Report from running a test suite."""

    suite: TestSuite
    results: list[TestResult]
    passed_count: int
    failed_count: int
    total_time_ms: float
    run_at: datetime = field(default_factory=datetime.now)

    @property
    def pass_rate(self) -> float:
        total = self.passed_count + self.failed_count
        return self.passed_count / total if total > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite.id,
            "suite_name": self.suite.name,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": round(self.pass_rate, 3),
            "total_time_ms": round(self.total_time_ms, 2),
            "run_at": self.run_at.isoformat(),
            "results": [r.to_dict() for r in self.results],
        }


class RetrievalTester:
    """
    Test retrieval quality before embedding and export.

    This is the killer feature of CHONK - allows users to:
    1. Index chunks locally with a fast embedding model
    2. Test queries and see which chunks are retrieved
    3. Save queries into test suites for regression testing
    4. Run test suites and get pass/fail reports
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize retrieval tester.

        Args:
            model_name: Embedding model to use for testing
        """
        self.embedder = Embedder(model_name)
        self._chunk_embeddings: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._chunk_to_doc: dict[str, tuple[str, str]] = {}  # chunk_id -> (doc_id, doc_name)
        self._is_indexed = False

    @property
    def is_indexed(self) -> bool:
        """Check if chunks are indexed and ready for search."""
        return self._is_indexed

    @property
    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self._chunks)

    def index_document(self, document: ChonkDocument) -> int:
        """
        Index a single document's chunks.

        Args:
            document: Document to index

        Returns:
            Number of chunks indexed
        """
        return self.index_documents([document])

    def index_documents(self, documents: list[ChonkDocument]) -> int:
        """
        Index chunks from multiple documents.

        Args:
            documents: Documents to index

        Returns:
            Total number of chunks indexed
        """
        # Gather all chunks
        self._chunks = []
        self._chunk_to_doc = {}

        for doc in documents:
            doc_name = doc.source_path.name if doc.source_path else doc.id
            for chunk in doc.chunks:
                self._chunks.append(chunk)
                self._chunk_to_doc[chunk.id] = (doc.id, doc_name)

        if not self._chunks:
            self._chunk_embeddings = np.array([])
            self._is_indexed = True
            return 0

        # Embed all chunks
        texts = [c.content for c in self._chunks]
        self._chunk_embeddings = self.embedder.embed_many(texts, show_progress=True)
        self._is_indexed = True

        return len(self._chunks)

    def index_project(self, project: ChonkProject) -> int:
        """
        Index all documents in a project.

        Args:
            project: Project to index

        Returns:
            Total number of chunks indexed
        """
        return self.index_documents(project.documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Search for chunks matching a query.

        Args:
            query: Search query text
            top_k: Number of results to return
            document_ids: Optional list of document IDs to restrict search

        Returns:
            List of search results sorted by relevance
        """
        if not self._is_indexed:
            raise ValueError("Must call index_document(s) before searching")

        if len(self._chunks) == 0:
            return []

        # Embed query
        query_embedding = self.embedder.embed(query)

        # Filter by document if specified
        if document_ids:
            mask = np.array([
                self._chunk_to_doc.get(c.id, ("", ""))[0] in document_ids
                for c in self._chunks
            ])
            if not mask.any():
                return []
            embeddings = self._chunk_embeddings[mask]
            chunks = [c for c, m in zip(self._chunks, mask) if m]
        else:
            embeddings = self._chunk_embeddings
            chunks = self._chunks

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, embeddings)

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = chunks[idx]
            doc_id, doc_name = self._chunk_to_doc.get(chunk.id, (None, None))
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(similarities[idx]),
                    rank=rank + 1,
                    document_id=doc_id,
                    document_name=doc_name,
                )
            )

        return results

    def run_test(
        self,
        query: TestQuery,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> TestResult:
        """
        Run a single test query and check results.

        Args:
            query: Test query with expected/excluded chunks
            top_k: Number of results to consider
            document_ids: Optional document filter

        Returns:
            TestResult with pass/fail status
        """
        import time

        start = time.time()
        results = self.search(query.query, top_k, document_ids)
        elapsed_ms = (time.time() - start) * 1000

        result_ids = {r.chunk.id for r in results}

        # Check expected chunks
        missing = [
            cid for cid in query.expected_chunk_ids
            if cid not in result_ids
        ]

        # Check excluded chunks
        unwanted = [
            cid for cid in query.excluded_chunk_ids
            if cid in result_ids
        ]

        passed = len(missing) == 0 and len(unwanted) == 0

        return TestResult(
            query=query,
            results=results,
            passed=passed,
            missing_expected=missing,
            unexpected_excluded=unwanted,
            execution_time_ms=elapsed_ms,
        )

    def run_test_suite(
        self,
        suite: TestSuite,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> TestSuiteReport:
        """
        Run all tests in a test suite.

        Args:
            suite: Test suite to run
            top_k: Number of results per query
            document_ids: Optional document filter

        Returns:
            TestSuiteReport with all results
        """
        import time

        start = time.time()
        results = []
        passed = 0
        failed = 0

        for query in suite.queries:
            result = self.run_test(query, top_k, document_ids)
            results.append(result)
            if result.passed:
                passed += 1
            else:
                failed += 1

        total_ms = (time.time() - start) * 1000

        return TestSuiteReport(
            suite=suite,
            results=results,
            passed_count=passed,
            failed_count=failed,
            total_time_ms=total_ms,
        )

    def get_chunk_coverage(
        self,
        suite: TestSuite,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Analyze which chunks are retrieved by test queries.

        Helps identify:
        - Chunks that are never retrieved
        - Chunks that are retrieved by many queries (potential duplicates)

        Args:
            suite: Test suite to analyze
            top_k: Number of results per query

        Returns:
            Coverage analysis
        """
        chunk_hits: dict[str, list[str]] = {c.id: [] for c in self._chunks}

        for query in suite.queries:
            results = self.search(query.query, top_k)
            for r in results:
                chunk_hits[r.chunk.id].append(query.id)

        never_retrieved = [
            cid for cid, hits in chunk_hits.items()
            if len(hits) == 0
        ]

        frequently_retrieved = [
            {"chunk_id": cid, "query_count": len(hits)}
            for cid, hits in chunk_hits.items()
            if len(hits) >= 3
        ]

        return {
            "total_chunks": len(self._chunks),
            "never_retrieved_count": len(never_retrieved),
            "never_retrieved": never_retrieved,
            "frequently_retrieved": frequently_retrieved,
            "coverage_rate": 1 - (len(never_retrieved) / len(self._chunks))
            if self._chunks else 0,
        }

    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute chunk-to-chunk similarity matrix.

        Useful for finding redundant or near-duplicate chunks.

        Returns:
            NxN similarity matrix where N is the number of chunks
        """
        if not self._is_indexed or self._chunk_embeddings is None:
            raise ValueError("Must index chunks first")

        n = len(self._chunks)
        if n == 0:
            return np.array([])

        # Compute pairwise cosine similarity
        norms = np.linalg.norm(self._chunk_embeddings, axis=1, keepdims=True)
        normalized = self._chunk_embeddings / (norms + 1e-10)
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix

    def find_similar_chunks(
        self,
        chunk_id: str,
        threshold: float = 0.8,
    ) -> list[tuple[Chunk, float]]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of chunk to compare against
            threshold: Minimum similarity score

        Returns:
            List of (chunk, similarity) tuples
        """
        # Find chunk index
        chunk_idx = None
        for i, c in enumerate(self._chunks):
            if c.id == chunk_id:
                chunk_idx = i
                break

        if chunk_idx is None:
            raise ValueError(f"Chunk not found: {chunk_id}")

        # Get similarities
        chunk_embedding = self._chunk_embeddings[chunk_idx]
        similarities = self._cosine_similarity(chunk_embedding, self._chunk_embeddings)

        # Find similar chunks (excluding self)
        results = []
        for i, sim in enumerate(similarities):
            if i != chunk_idx and sim >= threshold:
                results.append((self._chunks[i], float(sim)))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _cosine_similarity(
        self,
        query: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and embeddings."""
        query_norm = np.linalg.norm(query)
        embedding_norms = np.linalg.norm(embeddings, axis=1)

        # Avoid division by zero
        query_norm = max(query_norm, 1e-10)
        embedding_norms = np.maximum(embedding_norms, 1e-10)

        similarities = np.dot(embeddings, query) / (embedding_norms * query_norm)
        return similarities

    def clear_index(self) -> None:
        """Clear the current index."""
        self._chunk_embeddings = None
        self._chunks = []
        self._chunk_to_doc = {}
        self._is_indexed = False
