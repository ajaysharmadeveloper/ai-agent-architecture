"""Lesson 15, the shop the tools act on, and how long each tool takes.

Real tools wait on other systems: a courier's API, a database, a payment provider. Here each tool
sleeps for a fixed time to stand in for that wait. The waits are made up and written down below; the
timings in parallel.run.txt are what actually happened with them on this laptop.

The orders are an illustration. Which calls the model asked for, and in what order, is real.
"""
import threading
import time
from dataclasses import dataclass, field

TODAY = "2026-10-27"

# seconds each tool waits, standing in for the system behind it
WAIT = {
    "orders_get_order": 0.5,          # the shop's own database
    "shipping_track_package": 1.5,    # a courier's API
    "account_update_address": 0.8,    # writing the customer record
    "orders_ship_replacement": 1.0,   # booking a courier for a new parcel
}


@dataclass
class Order:
    order_id: str
    customer_id: str
    item: str
    status: str  # placed | shipped | delivered
    delivered_days_ago: int | None


ORDERS = {
    o.order_id: o
    for o in [
        Order("58213", "40917", "wireless headphones", "delivered", 18),
        Order("61145", "40917", "screen protector", "delivered", 11),
        Order("52210", "40917", "two coffee mugs", "delivered", 34),
        Order("64410", "18843", "running shoes", "shipped", None),
        Order("64411", "18843", "yoga mat", "shipped", None),
        Order("64412", "18843", "water bottle", "shipped", None),
    ]
}

TRACKING = {
    "64410": ("Bhiwandi sorting hub", "2026-10-29"),
    "64411": ("out for delivery, Andheri East", "2026-10-27"),
    "64412": ("Delhi air cargo hub", "2026-10-30"),
}

OLD_ADDRESS = "7 Lake View Apartments, Bengaluru 560034"


@dataclass
class World:
    """What actually happened, so the score is read off the world, not off the agent's reply."""
    address: dict = field(default_factory=lambda: {"40917": OLD_ADDRESS})
    shipped_to: list[tuple[str, str]] = field(default_factory=list)  # (order_id, address on the parcel)
    calls: list[str] = field(default_factory=list)
    batches: list[int] = field(default_factory=list)                # how many calls each model turn asked for
    tool_seconds: float = 0.0                                      # wall-clock time spent running tools
    lock: threading.Lock = field(default_factory=threading.Lock)

    def wait(self, tool: str) -> None:
        time.sleep(WAIT[tool])
