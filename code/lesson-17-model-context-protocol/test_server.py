"""Lesson 17, checks for the support MCP server. No model needed: the client talks to the server in-process.

    python -m pytest test_server.py -q
"""
import asyncio

from mcp import Client

import shop
from support_server import server


def run(coro):
    return asyncio.run(coro)


async def _tools():
    async with Client(server) as c:
        return (await c.list_tools()).tools


async def _call(name, args):
    async with Client(server) as c:
        return await c.call_tool(name, args)


def test_server_lists_the_three_support_tools():
    names = sorted(t.name for t in run(_tools()))
    assert names == ["orders_get_order", "payments_refund", "shipping_track_package"]


def test_order_id_schema_carries_the_lesson_13_description_and_pattern():
    schema = next(t for t in run(_tools()) if t.name == "orders_get_order").input_schema
    order_id = schema["properties"]["order_id"]
    assert order_id["pattern"] == "^[0-9]{5}$"
    assert "5-digit" in order_id["description"]


def test_a_wrong_call_reaches_the_model_with_its_reason():
    result = run(_call("shipping_track_package", {"order_id": "58213"}))
    assert result.is_error
    assert "delivered on 2026-10-18" in result.content[0].text


def test_refund_happens_once():
    shop.REFUNDED.clear()
    first = run(_call("payments_refund", {"order_id": "58213"}))
    second = run(_call("payments_refund", {"order_id": "58213"}))
    assert not first.is_error and second.is_error
    assert "already refunded" in second.content[0].text


def test_a_shipped_order_cannot_be_refunded():
    result = run(_call("payments_refund", {"order_id": "64412"}))
    assert result.is_error and "not been delivered" in result.content[0].text


def test_refund_policy_is_a_resource():
    async def read():
        async with Client(server) as c:
            return (await c.read_resource("shop://policy/refunds")).contents[0].text
    assert "30 days" in run(read())
