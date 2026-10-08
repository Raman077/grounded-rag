"""Runtime configuration, read from environment variables prefixed with ``RAG_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    environment: Literal["dev", "test", "prod"] = "dev"

    # Vector store. Accepts an http(s) URL, ":memory:", or a local directory path
    # (embedded Qdrant, handy for running without Docker).
    qdrant_location: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    # v2 carries named dense + sparse vectors; a v1 collection cannot serve hybrid
    # queries, so the name is bumped rather than migrated in place.
    collection_name: str = "corpus_v2"
    collection_alias: str = "corpus_live"

    # Embeddings
    embedding_provider: Literal["fastembed", "voyage", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    voyage_api_key: SecretStr | None = None

    # Sparse (lexical) embeddings — the other half of hybrid retrieval
    sparse_provider: Literal["bm25", "hashing"] = "bm25"
    sparse_model: str = "Qdrant/bm25"

    # Chunking
    chunk_max_tokens: int = Field(default=350, ge=50, le=2000)
    chunk_overlap_tokens: int = Field(default=40, ge=0, le=500)

    # Generation
    llm_provider: Literal["anthropic", "opencode"] = "anthropic"
    generation_model: str = "claude-sonnet-5"
    generation_effort: Literal["low", "medium", "high"] = "low"
    generation_thinking: Literal["disabled", "adaptive"] = "disabled"
    opencode_url: str = "http://127.0.0.1:4096"
    opencode_model: str = "openrouter/qwen/qwen3.8-27b:free"
    opencode_agent: str = "rag"
    opencode_password: SecretStr | None = None
    llm_timeout_seconds: float = 120.0

    # Retrieval / context
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    retrieval_candidates: int = Field(default=20, ge=1, le=200)
    # Candidates each arm of the hybrid query contributes before fusion. Larger than
    # retrieval_candidates on purpose: fusion can only rank what the arms proposed.
    hybrid_prefetch_limit: int = Field(default=50, ge=1, le=500)

    # Reranking — a cross-encoder reorders the candidate pool; see rag_core.rerank
    rerank_enabled: bool = True
    rerank_provider: Literal["cross-encoder", "none"] = "cross-encoder"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=50, ge=1, le=500)

    # Relevance gate. Applies to ScoredChunk.normalized_score (0-1), whatever produced
    # it. Fit with `python eval/harness/calibrate_gate.py`, not by hand — see rag_core.gate.
    min_relevance_score: float = 0.55
    strict_score_margin: float = 0.1
    max_context_tokens: int = 8000
    max_query_chars: int = 4000

    # Prompts
    prompt_dir: Path = Path("prompts")
    prompt_name: str = "answer-v1"

    # Auth
    jwt_secret: SecretStr = SecretStr("dev-insecure-secret-change-me-32bytes")
    jwt_algorithm: str = "HS256"
    jwt_audience: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
