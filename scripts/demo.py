"""Run the pipeline end to end on the NIST demo document and write demo/DEMO.md.

    python scripts/demo.py            # retrieval only (no API key needed)
    python scripts/demo.py --answer   # also generate Claude answers (needs ANTHROPIC_API_KEY)
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("RAG_INDEX_DIR", str(ROOT / "demo" / "index"))
sys.path.insert(0, str(ROOT / "src"))

from ragharness import config, harness  # noqa: E402
from ragharness.index import Index, build  # noqa: E402
from ragharness.ingest import ingest_pdf  # noqa: E402
from ragharness.retrieve import search  # noqa: E402

PDF = ROOT / "demo" / "nist.sp.800-61r3.pdf"
QUESTIONS = [
    ("ask", "What are the CSF 2.0 Functions and how do they map to incident response?"),
    ("ask", "Who should be involved in incident response besides the security team?"),
    ("ask", "What should an organization do after an incident to improve?"),
    ("ask", "How should incidents be prioritized?"),
    ("agent", "Compare the role of the Govern function with the Respond function in incident handling."),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answer", action="store_true", help="call Claude for answers")
    args = p.parse_args()

    if not (config.INDEX_DIR / "chunks.jsonl").exists():
        build(ingest_pdf(PDF))
    idx = Index()
    doc_tokens = sum(config.est_tokens(c.text) for c in idx.chunks)

    out = [
        "# Demo: NIST SP 800-61 Rev. 3",
        "",
        f"Source: [NIST SP 800-61r3](nist.sp.800-61r3.pdf), *Incident Response Recommendations and "
        f"Considerations for Cybersecurity Risk Management* (public domain, US Government).",
        "",
        f"Indexed into **{len(idx.chunks)} chunks** (~{doc_tokens:,} tokens). "
        f"Model: `{config.MODEL}`, effort `{config.EFFORT}`, context budget {config.CONTEXT_BUDGET} tokens.",
        "",
        "| # | Mode | Question | Context sent | vs. whole document | Cost |",
        "|---|------|----------|-------------:|-------------------:|-----:|",
    ]
    details, total_cost = [], 0.0
    for n, (mode, q) in enumerate(QUESTIONS, 1):
        t = time.time()
        hits = search(idx, q)
        ctx = sum(config.est_tokens(h.text) for h in hits)
        block = [f"## {n}. {q}", "", f"*Mode: `{mode}`*", ""]
        cost = "n/a"
        if args.answer:
            ans = harness.agent(idx, q) if mode == "agent" else harness.ask(idx, q, use_cache=False)
            total_cost += ans.usage.cost
            cost = f"${ans.usage.cost:.4f}"
            block += [ans.text, "", f"`{ans.usage.line()}` in {time.time() - t:.1f}s", ""]
        block += ["<details><summary>Retrieved sources</summary>", ""]
        block += [f"- **{h.path}** (p.{h.pages}), rerank score {h.score:.2f}" for h in hits]
        block += ["", "</details>", ""]
        details += block
        out.append(f"| {n} | {mode} | {q} | ~{ctx:,} tok | {ctx / doc_tokens:.0%} | {cost} |")

    if args.answer:
        out += ["", f"**Total cost for {len(QUESTIONS)} questions: ${total_cost:.4f}**"]
    else:
        out += ["", "*Answers not generated yet: run `python scripts/demo.py --answer` with an API key.*"]
    out += [""] + details
    (ROOT / "demo" / "DEMO.md").write_text("\n".join(out))
    print("\n".join(out[:16]))


if __name__ == "__main__":
    main()
