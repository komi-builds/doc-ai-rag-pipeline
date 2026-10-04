"""Harness logic with a fake Claude client and a fake retriever (offline)."""
from types import SimpleNamespace as NS

from ragharness import config, harness, llm
from ragharness.retrieve import Hit


def _usage():
    return NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)


class FakeClient:
    """Replays scripted responses and records each request."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(dict(kw, messages=list(kw["messages"])))
        return self.responses.pop(0)


def _text(t):
    return NS(content=[NS(type="text", text=t)], stop_reason="end_turn", usage=_usage())


def _tool(q, tid="t1"):
    return NS(content=[NS(type="tool_use", id=tid, input={"query": q})], stop_reason="tool_use", usage=_usage())


def _setup(monkeypatch, responses, tmp_path):
    fake = FakeClient(responses)
    monkeypatch.setattr(llm, "_client", fake)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(harness, "search", lambda idx, q, **kw: [Hit([0], "Ch 1 > Intro", "3", f"text for {q}", 1.0)])
    return fake


def test_ask_single_call_with_citable_context(monkeypatch, tmp_path):
    fake = _setup(monkeypatch, [_text("Answer [S1]")], tmp_path)
    ans = harness.ask(None, "What is CIA?")
    assert ans.text == "Answer [S1]" and ans.usage.calls == 1
    sent = fake.requests[0]["messages"][-1]["content"]
    assert '<source id="S1"' in sent and sent.endswith("Question: What is CIA?")
    assert fake.requests[0]["system"] == llm.SYSTEM  # frozen, cache-friendly


def test_ask_answer_cache_skips_second_call(monkeypatch, tmp_path):
    fake = _setup(monkeypatch, [_text("Answer")], tmp_path)
    harness.ask(None, "Q")
    again = harness.ask(None, "Q")
    assert again.usage.cached_answer and len(fake.requests) == 1


def test_agent_searches_then_answers(monkeypatch, tmp_path):
    fake = _setup(monkeypatch, [_tool("govern"), _text("Final [1:S1]")], tmp_path)
    ans = harness.agent(None, "Compare govern and respond")
    assert ans.text == "Final [1:S1]" and ans.usage.calls == 2
    last = fake.requests[1]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "user"]
    assert last[2]["content"][0]["type"] == "tool_result"


def test_agent_step_cap_forces_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AGENT_MAX_STEPS", 2)
    fake = _setup(monkeypatch, [_tool("a"), _tool("b"), _text("Done")], tmp_path)
    ans = harness.agent(None, "Q")
    assert ans.text == "Done" and len(fake.requests) == 3
    final_user = fake.requests[2]["messages"][-1]["content"]
    assert final_user[-1]["type"] == "text"  # limit notice rides in the same user turn


def test_refusal_is_reported(monkeypatch, tmp_path):
    refused = NS(content=[], stop_reason="refusal", usage=_usage())
    _setup(monkeypatch, [refused], tmp_path)
    assert "declined" in harness.ask(None, "Q", use_cache=False).text
