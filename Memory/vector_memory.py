# ╔══════════════════════════════════════════════════════════════════╗
# ║           PARMANA 2.0 — Vector Memory (Long-Term)               ║
# ║  ChromaDB + sentence-transformers for semantic recall.          ║
# ║  Persists across sessions. Auto-chunks long content.            ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ── Result Schema ─────────────────────────────────────────────────────────────

@dataclass
class MemoryResult:
    text: str
    score: float                   # cosine similarity 0-1 (higher = more relevant)
    doc_id: str
    source: str                    # "user" | "assistant" | "tool" | "injected"
    timestamp: float
    metadata: dict


# ── Vector Memory ─────────────────────────────────────────────────────────────

class VectorMemory:
    """
    Long-term semantic memory backed by ChromaDB (local, no server).

    Workflow:
        store()   → chunk + embed + upsert into Chroma
        search()  → embed query + cosine search → top-K results
        recall()  → search() formatted as injection string for prompt_manager
        forget()  → delete by doc_id or metadata filter
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        mem_cfg = cfg.get("memory", {}).get("vector", {})

        self._enabled = mem_cfg.get("enabled", True)
        self._persist_dir = os.path.expanduser(
            mem_cfg.get("persist_dir", "./data/chroma")
        )
        self._embedding_model_name = mem_cfg.get(
            "embedding_model", "all-MiniLM-L6-v2"
        )
        self._collection_name = mem_cfg.get("collection_name", "parmana_memory")
        self._top_k = mem_cfg.get("top_k", 5)
        self._score_threshold = mem_cfg.get("score_threshold", 0.45)
        self._chunk_size = mem_cfg.get("chunk_size", 512)
        self._chunk_overlap = mem_cfg.get("chunk_overlap", 64)

        self._client = None
        self._collection = None
        self._embedder = None

        if self._enabled:
            self._init_chroma()
            self._init_embedder()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        import chromadb
        from chromadb.config import Settings

        os.makedirs(self._persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            f"Chroma collection '{self._collection_name}' ready "
            f"({self._collection.count()} docs) at {self._persist_dir}"
        )

    def _init_embedder(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self._embedding_model_name)
        logger.debug(f"Embedder loaded: {self._embedding_model_name}")

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> list[str]:
        """
        Naive word-level chunking with overlap.
        Good enough for conversational turns; swap for tiktoken if needed.
        """
        words = text.split()
        if len(words) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self._chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self._chunk_size - self._chunk_overlap

        return chunks

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts, normalize_embeddings=True).tolist()

    @staticmethod
    def _make_id(text: str, salt: str = "") -> str:
        return hashlib.sha256(f"{salt}{text}".encode()).hexdigest()[:32]

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        text: str,
        source: str = "user",
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> list[str]:
        """
        Chunk, embed, and upsert text into the vector store.

        Returns list of stored doc IDs.
        """
        if not self._enabled:
            return []

        if not text or not text.strip():
            return []

        meta_base = {
            "source": source,
            "timestamp": time.time(),
            **(metadata or {}),
        }

        chunks = self._chunk(text)
        embeddings = self._embed(chunks)

        ids, docs, metas, embeds = [], [], [], []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cid = doc_id or self._make_id(chunk, salt=str(i))
            ids.append(cid)
            docs.append(chunk)
            metas.append({**meta_base, "chunk_index": i, "total_chunks": len(chunks)})
            embeds.append(emb)

        self._collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeds,
        )
        logger.debug(f"Stored {len(ids)} chunk(s) from source='{source}'")
        return ids

    def store_turn(
        self,
        role: str,
        content: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[str]:
        """Convenience wrapper for storing a conversation turn."""
        meta = {}
        if provider:
            meta["provider"] = provider
        if model:
            meta["model"] = model
        return self.store(content, source=role, metadata=meta)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_source: Optional[str] = None,
    ) -> list[MemoryResult]:
        """
        Semantic search. Returns results above score_threshold, best first.
        """
        if not self._enabled or not query.strip():
            return []

        k = top_k or self._top_k
        threshold = score_threshold if score_threshold is not None else self._score_threshold

        where = {"source": filter_source} if filter_source else None

        query_emb = self._embed([query])[0]

        try:
            results = self._collection.query(
                query_embeddings=[query_emb],
                n_results=min(k, max(1, self._collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

        output: list[MemoryResult] = []
        for doc, meta, dist, doc_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            # Chroma cosine distance → similarity
            score = 1.0 - dist
            if score < threshold:
                continue
            output.append(
                MemoryResult(
                    text=doc,
                    score=round(score, 4),
                    doc_id=doc_id,
                    source=meta.get("source", "unknown"),
                    timestamp=meta.get("timestamp", 0.0),
                    metadata=meta,
                )
            )

        output.sort(key=lambda r: r.score, reverse=True)
        return output

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> str:
        """
        Search and format results as a string ready for prompt injection.
        Returns empty string if nothing relevant found.
        """
        results = self.search(query, top_k=top_k, score_threshold=score_threshold)
        if not results:
            return ""

        lines = ["## Recalled from memory"]
        for r in results:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.timestamp))
            lines.append(f"- [{r.source} @ {ts}, relevance={r.score}] {r.text}")

        return "\n".join(lines)

    # ── Delete ────────────────────────────────────────────────────────────────

    def forget(self, doc_ids: list[str]) -> None:
        """Delete specific documents by ID."""
        if not self._enabled or not doc_ids:
            return
        self._collection.delete(ids=doc_ids)
        logger.debug(f"Deleted {len(doc_ids)} doc(s) from vector memory")

    def forget_by_source(self, source: str) -> None:
        """Delete all documents from a given source (e.g. 'user', 'tool')."""
        if not self._enabled:
            return
        self._collection.delete(where={"source": source})
        logger.debug(f"Deleted all docs with source='{source}'")

    def clear_all(self) -> None:
        """Nuke the entire collection. Irreversible."""
        if not self._enabled:
            return
        self._client.delete_collection(self._collection_name)
        self._init_chroma()
        logger.warning("Vector memory cleared entirely.")

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        if not self._enabled or self._collection is None:
            return 0
        return self._collection.count()

    def summary_line(self) -> str:
        return f"vector_memory: {self.count} chunks | model={self._embedding_model_name}"

    def __repr__(self) -> str:
        return f"<VectorMemory enabled={self._enabled} count={self.count}>"
