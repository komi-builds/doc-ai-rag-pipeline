"""The two answering modes.

ask():   retrieve once, then one Claude call. Cheapest; right for most questions.
agent(): Claude gets a search tool and decides what to look up. Costs more, but
         handles multi-part questions ("compare X and Y") that one search misses.
"""
import json

from . import config, llm
from .index import Index
from .retrieve import format_context, search

SEARCH_TOOL = {
    "name": "search_book",
    "description": (
        "Search the textbook. Returns the most relevant passages as <source> blocks with "
        "section paths and page numbers. Use short, specific queries; make one call per "
        "distinct concept rather than one broad query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to look up"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

AGENT_SYSTEM = llm.SYSTEM + """
- You have a search_book tool. Search before answering; search again only if a needed concept is missing.
- Source ids restart in each search result, so cite as [query-number:S#], e.g. [1:S2]."""


def _sources(hits):
    return [{"section": h.path, "pages": h.pages} for h in hits]


def ask(index: Index, question: str, history=None, use_cache=True) -> llm.Answer:
    hits = search(index, question)
    context = format_context(hits)
    cache = llm.AnswerCache()
    hist_key = json.dumps(history or [])
    if use_cache and (hit := cache.get(question, context, hist_key)):
        return llm.Answer(hit["text"], llm.Usage(cached_answer=True), hit["sources"])

    usage = llm.Usage()
    messages = list(history or []) + [
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
    ]
    answer = llm.text_of(llm.create(messages, usage))
    sources = _sources(hits)
    if use_cache:
        cache.put({"text": answer, "sources": sources}, question, context, hist_key)
    return llm.Answer(answer, usage, sources)


def agent(index: Index, question: str, verbose=False) -> llm.Answer:
    usage = llm.Usage()
    messages = [{"role": "user", "content": question}]
    sources, n_search = [], 0
    # Smaller per-search budget: the agent may search several times.
    budget = max(600, config.CONTEXT_BUDGET // 2)

    for step in range(config.AGENT_MAX_STEPS + 1):
        resp = llm.create(messages, usage, tools=[SEARCH_TOOL], system=AGENT_SYSTEM)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return llm.Answer(llm.text_of(resp), usage, sources)

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            n_search += 1
            query = block.input.get("query", "") if isinstance(block.input, dict) else ""
            if verbose:
                print(f"  search {n_search}: {query}")
            hits = search(index, query, top_k=4, budget=budget)
            sources += [dict(s, query=n_search) for s in _sources(hits)]
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"[search {n_search}]\n{format_context(hits)}",
            })
        if step == config.AGENT_MAX_STEPS - 1:
            results.append({"type": "text", "text": "Search limit reached. Answer now from these sources."})
        messages.append({"role": "user", "content": results})

    return llm.Answer(llm.text_of(resp), usage, sources)
