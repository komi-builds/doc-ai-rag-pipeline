"""Claude calls with token accounting, an on-disk answer cache, and refusal fallbacks."""
import hashlib
import json
from dataclasses import dataclass, field

import anthropic

from . import config

# Frozen system prompt: never interpolate per-request values here, or prompt caching breaks.
SYSTEM = """You are a precise study assistant answering questions from a textbook.

Rules:
- Use only the provided <source> blocks. If they do not contain the answer, say so in one sentence.
- Cite sources inline as [S1], [S2] after the claims they support.
- Be concise: lead with the direct answer, then at most a few supporting sentences or bullets.
- Prefer the textbook's terminology, since the reader is preparing for an exam."""

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0
    cached_answer: bool = False
    model: str = config.MODEL

    def add(self, u):
        self.input += u.input_tokens or 0
        self.output += u.output_tokens or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.calls += 1

    @property
    def cost(self):
        p_in, p_out, p_read, p_write = config.PRICES.get(self.model, config.PRICES["claude-sonnet-5-5"])
        return (self.input * p_in + self.output * p_out + self.cache_read * p_read
                + self.cache_write * p_write) / 1e6

    def line(self):
        if self.cached_answer:
            return "answer cache hit: 0 tokens, $0.0000"
        return (f"{self.calls} call(s) | in {self.input} (+{self.cache_read} cached) | "
                f"out {self.output} | ${self.cost:.4f}")


@dataclass
class Answer:
    text: str
    usage: Usage = field(default_factory=Usage)
    sources: list = field(default_factory=list)


def create(messages, usage: Usage, tools=None, system=SYSTEM):
    """One Messages API call. Server-side fallbacks retry on another model if a
    safety classifier declines, which matters for security course material."""
    kwargs = dict(
        model=config.MODEL,
        max_tokens=config.MAX_TOKENS,
        system=system,
        messages=messages,
        output_config={"effort": config.EFFORT},
        cache_control={"type": "ephemeral"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if tools:
        kwargs["tools"] = tools
    resp = client().beta.messages.create(**kwargs)
    usage.add(resp.usage)
    return resp


def text_of(resp):
    if resp.stop_reason == "refusal":
        return "The model declined to answer this question."
    out = "".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason == "max_tokens":
        out += "\n[answer truncated: raise RAG_MAX_TOKENS]"
    return out


class AnswerCache:
    """Identical question + identical context = identical answer, so never pay twice."""

    def __init__(self):
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, *parts):
        key = hashlib.sha256("\x1f".join([config.MODEL, config.EFFORT, *parts]).encode()).hexdigest()[:24]
        return config.CACHE_DIR / f"{key}.json"

    def get(self, *parts):
        p = self._path(*parts)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, value, *parts):
        self._path(*parts).write_text(json.dumps(value))
