"""Retrieve -> fuse -> rerank -> compress -> pack into a fixed token budget.

Each step either improves accuracy (fusion, reranking) or cuts tokens
(merging neighbors, sentence-level compression, the budget cap).
"""
import re
from dataclasses import dataclass

import numpy as np

from . import config
from .index import Index, tokenize

_reranker = None


def reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(config.RERANK_MODEL)
    return _reranker


@dataclass
class Hit:
    chunk_ids: list
    path: str
    pages: str
    text: str
    score: float


def rrf(rankings, k=60):
    """Reciprocal rank fusion: robust way to merge rankings with incomparable scores."""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def compress(text, query, max_tokens):
    """Keep the sentences most related to the query, in original order."""
    if config.est_tokens(text) <= max_tokens:
        return text
    sents = re.split(r"(?<=[.!?:])\s+|\n", text)
    q = set(tokenize(query))
    scored = [(len(q & set(tokenize(s))) + (0.5 if i == 0 else 0), i) for i, s in enumerate(sents)]
    keep, used = set(), 0
    for score, i in sorted(scored, key=lambda x: (-x[0], x[1])):
        cost = config.est_tokens(sents[i])
        if used + cost > max_tokens:
            continue
        keep.add(i)
        used += cost
    out, prev = [], -2
    for i in sorted(keep):
        if i != prev + 1 and out:
            out.append("...")
        out.append(sents[i].strip())
        prev = i
    return " ".join(out)


def _merge_neighbors(index, ids):
    """Join adjacent chunks from the same section so context is not repeated or cut mid-thought."""
    groups = []
    for i in sorted(ids):
        if groups and i == groups[-1][-1] + 1 and index.chunks[i].path == index.chunks[groups[-1][-1]].path:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def search(index: Index, query: str, top_k=None, budget=None, rerank=True):
    top_k = top_k or config.TOP_K
    budget = budget or config.CONTEXT_BUDGET
    n = config.CANDIDATES
    cand = rrf([index.lexical(query, n), index.dense(query, n)])[:n]

    if rerank:
        docs = [f"{index.chunks[i].path}\n{index.chunks[i].text}" for i in cand]
        scores = np.array(list(reranker().rerank(query, docs)))
        order = np.argsort(-scores)
        ranked = [(cand[j], float(scores[j])) for j in order[:top_k]]
    else:
        ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(cand[:top_k])]

    best = dict(ranked)
    hits = []
    for group in _merge_neighbors(index, best):
        cs = [index.chunks[i] for i in group]
        text = "\n".join(c.text for c in cs)
        pages = f"{cs[0].page_start}" if cs[0].page_start == cs[-1].page_end else f"{cs[0].page_start}-{cs[-1].page_end}"
        hits.append(Hit(group, cs[0].path, pages, text, max(best[i] for i in group)))
    hits.sort(key=lambda h: -h.score)

    # Spend the budget on the best hits first; compress each to its fair share.
    packed, used = [], 0
    for rank, h in enumerate(hits):
        remaining = budget - used
        if remaining < 60:
            break
        share = max(60, remaining // (len(hits) - rank))
        if rank == 0:
            share = max(share, min(remaining, budget // 2))
        h.text = compress(h.text, query, share)
        used += config.est_tokens(h.text) + config.est_tokens(h.path) + 10
        packed.append(h)
    return packed


def format_context(hits):
    """Compact tagged blocks: cheap to send, easy for Claude to cite as [S1], [S2]."""
    return "\n\n".join(
        f'<source id="S{n}" section="{h.path}" pages="{h.pages}">\n{h.text}\n</source>'
        for n, h in enumerate(hits, 1)
    )
