"""Central settings. Override any value with an environment variable of the same name."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _env(name, default, cast=str):
    val = os.environ.get(name)
    return cast(val) if val is not None else default


# Storage
INDEX_DIR = Path(_env("RAG_INDEX_DIR", str(ROOT / "index")))
CACHE_DIR = Path(_env("RAG_CACHE_DIR", str(ROOT / ".rag_cache")))

# Chunking (token counts are estimates: ~4 characters per token)
CHUNK_TOKENS = _env("RAG_CHUNK_TOKENS", 350, int)
CHUNK_OVERLAP_TOKENS = _env("RAG_CHUNK_OVERLAP_TOKENS", 50, int)

# Retrieval
EMBED_MODEL = _env("RAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANK_MODEL = _env("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
CANDIDATES = _env("RAG_CANDIDATES", 20, int)       # fused candidates sent to the reranker
TOP_K = _env("RAG_TOP_K", 6, int)                  # chunks kept after reranking
CONTEXT_BUDGET = _env("RAG_CONTEXT_BUDGET", 2000, int)  # max tokens of context sent to Claude

# Generation
MODEL = _env("RAG_MODEL", "claude-sonnet-5-5")
EFFORT = _env("RAG_EFFORT", "low")
MAX_TOKENS = _env("RAG_MAX_TOKENS", 4000, int)
AGENT_MAX_STEPS = _env("RAG_AGENT_MAX_STEPS", 4, int)

# USD per million tokens: (input, output, cache read, cache write 5m)
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
}


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)
