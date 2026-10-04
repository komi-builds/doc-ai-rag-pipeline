"""PDF -> structure-aware chunks.

The PDF's table of contents splits the book into real sections. Each chunk keeps
its breadcrumb (chapter > section > subsection) and page range, which helps
retrieval and lets answers cite their source.
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF

from . import config

SKIP_TITLES = {"front cover", "copyright", "contents", "index", "about the authors"}


@dataclass
class Chunk:
    id: int
    doc: str
    path: str        # "Chapter 2: Domain 1 > Cornerstone Concepts > Integrity"
    page_start: int  # 1-based PDF page
    page_end: int
    text: str


def _clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # rejoin hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_text(doc, pno: int) -> str:
    """Page text without running headers/footers (top and bottom 5% of the page)."""
    page = doc[pno]
    h = page.rect.height
    blocks = page.get_text("blocks")
    keep = [b[4] for b in sorted(blocks, key=lambda b: (b[1], b[0]))
            if b[6] == 0 and b[1] > h * 0.05 and b[3] < h * 0.95]
    return "\n".join(keep)


def _sections(doc):
    """Yield (breadcrumb, first_page, last_page, title) from the TOC (0-based pages)."""
    toc = [t for t in doc.get_toc() if t[2] > 0]
    stack = []
    for i, (lvl, title, page) in enumerate(toc):
        stack = stack[: lvl - 1] + [title.strip()]
        nxt = next((t[2] for t in toc[i + 1:]), doc.page_count + 1)
        yield " > ".join(stack), page - 1, max(page - 1, nxt - 1), title.strip(), toc[i + 1] if i + 1 < len(toc) else None


def _section_text(doc, title, first, last, next_entry) -> str:
    """Text from this heading up to the next TOC heading."""
    parts = [_page_text(doc, p) for p in range(first, min(last, doc.page_count - 1) + 1)]
    text = "\n".join(parts)
    start = text.find(title)
    if start > 0:
        text = text[start:]
    if next_entry is not None and next_entry[2] - 1 <= last:
        end = text.find(next_entry[1].strip(), len(title))
        if end > 0:
            text = text[:end]
    return _clean(text)


def _split(text: str, size: int, overlap: int):
    """Split on paragraph then sentence boundaries into ~size-token windows."""
    max_chars, ov_chars = size * 4, overlap * 4
    units = []
    for para in re.split(r"\n\s*\n", text):
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(re.split(r"(?<=[.!?])\s+", para))
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) > max_chars:
            chunks.append(cur.strip())
            cur = cur[-ov_chars:] if ov_chars else ""
            cur = cur[cur.find(" ") + 1:] if " " in cur else cur
        cur += ("\n" if cur else "") + u
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _locate(piece, pages, first, last):
    """Narrow a chunk to the exact pages its first and last lines appear on."""
    head, tail = _squash(piece[:60]), _squash(piece[-60:])
    span = range(first, last + 1)
    start = next((p for p in span if head in pages[p]), first)
    end = next((p for p in span if p >= start and tail in pages[p]), last)
    return start + 1, end + 1


def ingest_pdf(pdf_path: Path, start_id: int = 0):
    doc = fitz.open(pdf_path)
    name = pdf_path.stem
    pages = [_squash(_clean(_page_text(doc, p))) for p in range(doc.page_count)]
    chunks, cid = [], start_id
    for path, first, last, title, nxt in _sections(doc):
        if title.lower() in SKIP_TITLES:
            continue
        text = _section_text(doc, title, first, last, nxt)
        if len(text) < 80:  # heading with no body of its own
            continue
        last = min(last, doc.page_count - 1)
        for piece in _split(text, config.CHUNK_TOKENS, config.CHUNK_OVERLAP_TOKENS):
            chunks.append(Chunk(cid, name, path, *_locate(piece, pages, first, last), piece))
            cid += 1
    return chunks


def save_chunks(chunks, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: Path):
    with path.open() as f:
        return [Chunk(**json.loads(line)) for line in f]
