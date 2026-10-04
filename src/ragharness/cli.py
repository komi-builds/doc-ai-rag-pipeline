"""Command line entry point: rag ingest | search | ask | chat | eval"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from . import config  # noqa: E402


def cmd_ingest(args):
    from .index import build
    from .ingest import ingest_pdf

    chunks, t = [], time.time()
    for pdf in args.pdfs:
        part = ingest_pdf(Path(pdf), start_id=len(chunks))
        print(f"{pdf}: {len(part)} chunks")
        chunks += part
    total = sum(config.est_tokens(c.text) for c in chunks)
    print(f"Embedding {len(chunks)} chunks (~{total:,} tokens)...")
    build(chunks)
    print(f"Index written to {config.INDEX_DIR} in {time.time() - t:.0f}s")


def cmd_search(args):
    from .index import Index
    from .retrieve import search

    for h in search(Index(), args.query, top_k=args.k, rerank=not args.no_rerank):
        print(f"\n[{h.score:.2f}] {h.path} (p.{h.pages})\n{h.text[:500]}")


def _print_answer(ans, show_sources=True):
    print(f"\n{ans.text}\n")
    if show_sources and ans.sources:
        for n, s in enumerate(ans.sources, 1):
            tag = f"{s['query']}:S" if "query" in s else "S"
            num = n if "query" not in s else ""
            print(f"  [{tag}{num}] {s['section']} (p.{s['pages']})")
    print(f"\n  {ans.usage.line()}")


def cmd_ask(args):
    from . import harness
    from .index import Index

    idx = Index()
    if args.agent:
        ans = harness.agent(idx, args.question, verbose=True)
    else:
        ans = harness.ask(idx, args.question, use_cache=not args.no_cache)
    _print_answer(ans)


def cmd_chat(args):
    from . import harness
    from .index import Index

    idx = Index()
    history, total = [], 0.0
    print("Ask questions about your documents. Prefix with '/agent ' for agent mode. Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nSession cost: ${total:.4f}")
            return
        if not q:
            continue
        if q.startswith("/agent "):
            ans = harness.agent(idx, q[7:], verbose=True)
        else:
            ans = harness.ask(idx, q, history=history[-4:])
            # History keeps questions and answers only; old retrieved context is dropped.
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.text}]
        total += ans.usage.cost
        _print_answer(ans)


def cmd_eval(args):
    """Retrieval accuracy: does a retrieved source cover a page holding the answer?"""
    from .index import Index
    from .retrieve import search

    idx = Index()
    rows = [json.loads(line) for line in Path(args.file).open()]
    modes = {"bm25": None, "hybrid": False, "hybrid+rerank": True}
    for name, rerank in modes.items():
        hit1 = hitk = 0
        tokens = 0
        t = time.time()
        for r in rows:
            want = set(r["pages"])
            if rerank is None:
                ids = idx.lexical(r["q"], config.TOP_K)
                ranges = [range(idx.chunks[i].page_start, idx.chunks[i].page_end + 1) for i in ids]
                tokens += sum(config.est_tokens(idx.chunks[i].text) for i in ids)
            else:
                hits = search(idx, r["q"], rerank=rerank)
                ranges = []
                for h in hits:
                    a, _, b = h.pages.partition("-")
                    ranges.append(range(int(a), int(b or a) + 1))
                tokens += sum(config.est_tokens(h.text) for h in hits)
            found = [bool(want & set(rg)) for rg in ranges]
            hit1 += bool(found and found[0])
            hitk += any(found)
        n = len(rows)
        print(f"{name:14s} hit@1 {hit1 / n:6.1%}   hit@{config.TOP_K} {hitk / n:6.1%}   "
              f"avg context ~{tokens // n} tok   {(time.time() - t) / n:.2f}s/q")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag", description="Token-efficient RAG harness for Claude")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="Index one or more PDFs")
    s.add_argument("pdfs", nargs="+")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("search", help="Show retrieved passages (no API call)")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=config.TOP_K)
    s.add_argument("--no-rerank", action="store_true")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="Answer one question")
    s.add_argument("question")
    s.add_argument("--agent", action="store_true", help="let Claude run its own searches")
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("chat", help="Interactive session")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("eval", help="Measure retrieval accuracy (no API calls)")
    s.add_argument("file", nargs="?", default=str(config.ROOT / "tests" / "eval_questions.jsonl"))
    s.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
