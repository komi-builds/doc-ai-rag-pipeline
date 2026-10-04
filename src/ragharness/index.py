"""Hybrid index: BM25 (exact terms like "802.1X" or "CHAP") plus dense embeddings (meaning).

Everything runs locally, so indexing costs zero API tokens.
"""
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .ingest import load_chunks, save_chunks

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the
this to was were what when where which who why will with does do can should""".split())


def tokenize(text: str):
    toks = re.findall(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*", text.lower())
    return [t for t in toks if t not in STOP]


def _bm25_doc(c):
    # Heading words count twice: section titles are strong topical signals.
    title = c.path.split(" > ")[-1]
    return tokenize(f"{title} {title} {c.path} {c.text}")


_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(config.EMBED_MODEL)
    return _embedder


def build(chunks, index_dir=config.INDEX_DIR):
    index_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(chunks, index_dir / "chunks.jsonl")
    bm25 = BM25Okapi([_bm25_doc(c) for c in chunks])
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    passages = [f"{c.path}\n{c.text}" for c in chunks]
    vecs = np.array(list(embedder().embed(passages, batch_size=32)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(index_dir / "vectors.npy", vecs)


class Index:
    def __init__(self, index_dir=config.INDEX_DIR):
        if not (index_dir / "chunks.jsonl").exists():
            raise SystemExit(f"No index at {index_dir}. Run: rag ingest <pdf>")
        self.chunks = load_chunks(index_dir / "chunks.jsonl")
        with (index_dir / "bm25.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)
        self.vecs = np.load(index_dir / "vectors.npy")

    def lexical(self, query, n):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def dense(self, query, n):
        q = np.array(next(iter(embedder().query_embed(query))), dtype=np.float32)
        q /= np.linalg.norm(q)
        return list(np.argsort(-(self.vecs @ q))[:n])
