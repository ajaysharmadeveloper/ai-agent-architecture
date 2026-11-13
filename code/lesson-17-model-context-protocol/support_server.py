"""Lesson 17, the support tools as an MCP server. Written once; any MCP client can use it.

Run it on its own (it waits for a client on stdin/stdout):
    python support_server.py

Plug it into Claude Code, no code change:
    claude mcp add support -- python /full/path/to/support_server.py

Our own agent (agent.py) starts it the same way and talks to it over stdio.

Three kinds of thing a server can offer, and who decides when each one is used:
    tools      the MODEL decides to call them
    resources  the APPLICATION decides to read them and put them in front of the model
    prompts    the USER picks them, like a slash command
"""
from typing import Annotated

from pydantic import Field
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import shop

server = MCPServer("support")

OrderId = Annotated[str, Field(pattern=r"^[0-9]{5}$", description="The 5-digit order number, like 58213.")]


def _run(fn, order_id: str) -> dict:
    # Lesson 16: a call that can never work as asked goes back to the model WITH the reason.
    # A ToolError reaches the model as is_error plus this text. Any other exception reaches it as a bare
    # "Error executing tool", which tells the model nothing.
    try:
        return fn(order_id)
    except shop.ShopError as e:
        raise ToolError(str(e)) from e


@server.tool()
def orders_get_order(order_id: OrderId) -> dict:
    """Look up one order by its order ID: the item, the price, the status (shipped or delivered) and the
    delivery date. Changes nothing."""
    return _run(shop.get_order, order_id)


@server.tool()
def shipping_track_package(order_id: OrderId) -> dict:
    """Get where a shipped order's package is right now. Changes nothing. Use this when the customer asks
    where an order is. A delivered order has nothing to track."""
    return _run(shop.track_package, order_id)


@server.tool()
def payments_refund(order_id: OrderId) -> dict:
    """Refund one delivered order in full to the customer's original payment method. Moves money. Call it
    once per order, and only after the customer asked for a refund."""
    return _run(shop.refund, order_id)


@server.resource("shop://policy/refunds", name="refund_policy", mime_type="text/plain")
def refund_policy() -> str:
    """The shop's refund rules: what can be refunded, when, and how long the money takes."""
    return shop.REFUND_POLICY


@server.prompt()
def refund_request(order_id: str) -> str:
    """Start a refund conversation for one order."""
    return (f"A customer wants a refund for order {order_id}. Look the order up first, check it against the "
            f"refund policy, and only refund it if the policy allows.")


if __name__ == "__main__":
    server.run()   # stdio by default
