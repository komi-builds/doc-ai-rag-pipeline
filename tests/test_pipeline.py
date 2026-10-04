"""Offline tests: no API key or model downloads needed."""
from ragharness import config
from ragharness.index import tokenize
from ragharness.ingest import _split
from ragharness.retrieve import compress, rrf


def test_tokenize_keeps_technical_terms():
    toks = tokenize("What is 802.1X and how does PBKDF2 compare to SHA-256?")
    assert "802.1x" in toks and "pbkdf2" in toks and "sha-256" in toks
    assert "what" not in toks and "is" not in toks


def test_split_respects_size_and_covers_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(10))
    chunks = _split(text, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(config.est_tokens(c) <= 260 for c in chunks)
    assert "Paragraph 9" in chunks[-1]


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    assert fused[0] == 1 and set(fused) == {1, 2, 3, 4}


def test_compress_keeps_relevant_sentences_within_budget():
    text = ("Intro sentence about security. " + "Filler about unrelated topics here. " * 30
            + "CHAP uses a challenge and a shared secret. " + "More filler text. " * 30)
    out = compress(text, "how does CHAP work", max_tokens=40)
    assert "CHAP uses a challenge" in out
    assert config.est_tokens(out) <= 45


def test_compress_noop_when_under_budget():
    assert compress("Short text.", "query", 100) == "Short text."
