"""Lesson 8, the CODE EXECUTION version with smolagents: many actions in one trip.

Same two tools. The model writes Python that calls them as ordinary functions, and smolagents
runs that Python in a Docker container, not in this process. Only what the code prints comes back.

Tools are sent INTO the container, so each one is self-contained (its imports live inside it) and
reaches the data over HTTP: start `python billing_api.py` first.
"""
from smolagents import CodeAgent, OpenAIServerModel, tool


@tool
def list_invoices(month: str) -> list:
    """List every invoice due in a month, paid or not.

    Args:
        month: The month as YYYY-MM.
    """
    import json
    import urllib.request

    with urllib.request.urlopen(f"http://host.docker.internal:8765/invoices?month={month}") as r:
        return json.loads(r.read())


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send one email to one customer.

    Args:
        to: The customer's email address.
        subject: The subject line.
        body: The message.
    """
    import json
    import urllib.request

    req = urllib.request.Request(
        "http://host.docker.internal:8765/email",
        data=json.dumps({"to": to, "subject": subject, "body": body}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    with CodeAgent(
        tools=[list_invoices, send_email],
        model=OpenAIServerModel(model_id="gpt-4.1-mini"),
        executor_type="docker",  # the line that matters: the model's code runs in a sealed container
    ) as agent:
        agent.run(
            "Find every overdue invoice this month (2026-09) that is over Rs 50,000 "
            "and email each of those customers a reminder. Print how many went out."
        )
