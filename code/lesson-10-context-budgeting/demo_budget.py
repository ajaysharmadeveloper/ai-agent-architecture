"""Lesson 10, the support agent's twenty steps, run offline: no budget versus a budget.

Every step the customer says something, the agent calls lookup_order, and the tool sends back a
big result. Without a budget everything is kept and the input grows every step. With a budget the
input stays inside its envelopes. Run: python demo_budget.py
"""
from context_budget import BUDGET, build_step, estimate_tokens, format_report

RULES = "You are a support agent for an online store. Refunds over 5000 rupees need a human. " * 12
TOOLS = "lookup_order(order_id): returns the order, its items and its tracking history. " * 50
NOTES = "contact: message only\naddress: flat B\npromise: replace 58213 if not delivered by Friday"
DOCS = [
    {"score": 0.91, "text": "Refund policy: replacements ship within 5 working days. " * 60},
    {"score": 0.84, "text": "Delivery policy: late orders can be replaced once. " * 60},
    {"score": 0.62, "text": "Gift cards: gift cards cannot be refunded. " * 60},
    {"score": 0.40, "text": "Careers: we are hiring warehouse staff. " * 60},
]
# The tool returns the whole order history every time: the mistake this lesson warns about.
ORDER_DUMP = '{"order": 58213, "status": "in transit", "scan": "Pune hub"} ' * 160


def run(steps: int = 20):
    turns, results, unbudgeted = [], [], []
    for step in range(1, steps + 1):
        turns.append([
            {"role": "user", "content": f"Step {step}: any update on order 58213? " * 6},
            {"role": "assistant", "content": f"Checking order 58213 for you, step {step}. " * 6},
        ])
        results.append({"tool": "lookup_order", "text": ORDER_DUMP})
        everything = RULES + TOOLS + NOTES + "".join(d["text"] for d in DOCS)
        everything += "".join(r["text"] for r in results) + "".join(m["content"] for t in turns for m in t)
        unbudgeted.append(estimate_tokens(everything))
    system, messages, report = build_step(RULES, TOOLS, NOTES, DOCS, results, turns, "When will my replacement ship?")
    return unbudgeted, report, system, messages


if __name__ == "__main__":
    unbudgeted, report, _, messages = run()
    print("input tokens with NO budget, by step:")
    for step in (1, 5, 10, 20):
        print(f"  step {step:>2}: {unbudgeted[step - 1]:>7}")
    print("\nstep 20 WITH the budget:")
    print(format_report(report))
    print(f"\nthe question is the last thing the model reads: {messages[-1]['content'][-40:]!r}")
