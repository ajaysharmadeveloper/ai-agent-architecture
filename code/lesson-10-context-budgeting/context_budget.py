"""Lesson 10, context budgeting. The pure parts: no model calls, no SDK, so they run and test offline.

The context window is the model's working memory for ONE step. On every call your code decides what
goes into it. This file gives each kind of content (a "slot") an allowance, fills the fixed slots
first, trims the flexible ones to fit, puts the rules at the start and the question at the end, and
reports what each slot spent, the way Claude Code's /context does.

estimate_tokens() is a deliberately rough stand-in so this runs offline. Before you rely on a number,
count the real request with the token counting endpoint: client.messages.count_tokens(...).
"""

# What we allow ourselves for one step: far under the model's maximum, on purpose.
STEP_BUDGET = 32_000

# The envelopes. The reply is reserved first, then the fixed slots, then the flexible ones.
BUDGET = {
    "reply": 2_000,  # room for the answer, reserved before anything else
    "rules": 1_500,  # the system prompt's rules
    "tools": 3_000,  # tool definitions
    "notes": 500,  # pinned facts (lesson 8) and long-term notes (lesson 9)
    "docs": 6_000,  # retrieved policy passages
    "tool_results": 6_000,  # what the tools sent back
    "history": 8_000,  # the conversation so far
}
FIXED = ("rules", "tools", "notes")


def estimate_tokens(text: str) -> int:
    """Rough: about four characters per token of English. Use count_tokens for the real number."""
    return max(1, len(text) // 4)


def check_budget(budget: dict, step_budget: int = STEP_BUDGET) -> None:
    total = sum(budget.values())
    if total > step_budget:
        raise ValueError(f"allowances add up to {total}, more than the step budget of {step_budget}")


def check_fixed(name: str, text: str, allowance: int) -> int:
    """A fixed slot that does not fit is a bug in your prompt, not something to trim silently."""
    used = estimate_tokens(text)
    if used > allowance:
        raise ValueError(f"fixed slot '{name}' uses {used} tokens but its allowance is {allowance}")
    return used


def fit_docs(docs: list[dict], allowance: int, min_score: float = 0.5) -> list[dict]:
    """Best-scored passages first, whole passages only, until the envelope is full.
    A passage that barely matches is noise even when it fits, so it never gets in."""
    kept, used = [], 0
    for doc in sorted(docs, key=lambda d: d["score"], reverse=True):
        if doc["score"] < min_score:
            break
        cost = estimate_tokens(doc["text"])
        if used + cost <= allowance:
            kept.append(doc)
            used += cost
    return kept


def shrink_result(result: dict, allowance: int) -> dict:
    """One result bigger than the whole envelope keeps its head and says where the rest lives."""
    cost = estimate_tokens(result["text"])
    if cost <= allowance:
        return result
    head = result["text"][: allowance * 4 - 200]
    note = f"\n[cut {cost - estimate_tokens(head)} tokens; call {result['tool']} again with a narrower query for the rest]"
    return {**result, "text": head + note}


def fit_tool_results(results: list[dict], allowance: int, keep: int = 3) -> list[dict]:
    """Keep the newest `keep` results in full. Older ones become a one-line placeholder,
    so the model still knows the call happened without paying for its output again."""
    newest = results[-keep:]
    cleared = [{**r, "text": f"[cleared: an older result of {r['tool']}]"} for r in results[:-keep]]
    # The placeholders cost tokens too, so they come out of the envelope first.
    left = allowance - sum(estimate_tokens(r["text"]) for r in cleared)
    fitted = cleared + [shrink_result(r, left // len(newest)) for r in newest]
    used = sum(estimate_tokens(r["text"]) for r in fitted)
    if used > allowance:
        raise ValueError(f"tool results still use {used} tokens after clearing; lower `keep`")
    return fitted


def fit_history(turns: list[list[dict]], allowance: int) -> list[list[dict]]:
    """Newest whole turns that fit. A turn is never split, so a tool call never loses its result."""
    kept, used = [], 0
    for turn in reversed(turns):
        cost = sum(estimate_tokens(m["content"]) for m in turn)
        if used + cost > allowance:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    return kept


def build_step(rules, tools_text, notes, docs, tool_results, turns, question, budget=BUDGET):
    """Build one call's context to fit the budget. Returns (system, messages, report).

    Placement: rules and notes go FIRST (the system prompt), the bulky middle holds the history,
    the documents and the tool results, and the user's question goes LAST.
    """
    check_budget(budget)
    report = {name: check_fixed(name, text, budget[name]) for name, text in (("rules", rules), ("tools", tools_text), ("notes", notes))}

    kept_docs = fit_docs(docs, budget["docs"])
    kept_results = fit_tool_results(tool_results, budget["tool_results"])
    kept_turns = fit_history(turns, budget["history"])

    report["docs"] = sum(estimate_tokens(d["text"]) for d in kept_docs)
    report["tool_results"] = sum(estimate_tokens(r["text"]) for r in kept_results)
    report["history"] = sum(estimate_tokens(m["content"]) for t in kept_turns for m in t)

    system = rules + "\n\nWhat you know about this customer:\n" + notes
    messages = [m for t in kept_turns for m in t]
    middle = "\n\n".join([d["text"] for d in kept_docs] + [r["text"] for r in kept_results])
    messages.append({"role": "user", "content": middle + "\n\nCustomer's question: " + question})
    return system, messages, report


def format_report(report: dict, budget: dict = BUDGET) -> str:
    """A text version of Claude Code's /context: what each slot spent against its allowance."""
    lines = [f"{'slot':<13}{'used':>7}{'allowed':>9}"]
    for name in budget:
        if name == "reply":
            continue
        used, allowed = report[name], budget[name]
        bar = "#" * round(20 * used / allowed)
        lines.append(f"{name:<13}{used:>7}{allowed:>9}  {bar}")
    total = sum(report.values())
    lines.append(f"{'input total':<13}{total:>7}{STEP_BUDGET - budget['reply']:>9}  (reply reserved: {budget['reply']})")
    return "\n".join(lines)
