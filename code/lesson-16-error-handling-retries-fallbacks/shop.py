"""Lesson 16, the shop and the two outside services its tools depend on: a courier and a payment provider.

Real services fail. Here each failure is planted on purpose so every run hits the same one:

    busy        the courier answers "503 busy" twice, then works (a transient failure)
    down        the courier answers "503 busy" every time (an outage)
    lost_reply  the payment provider pays the refund, then the reply times out (a write that succeeded)

The orders, prices and waits are an illustration. What the model asked for, what the code retried
and where the money went are what actually happened, and they are in errors.run.txt.
"""
import time
from dataclasses import dataclass, field

TODAY = "2026-11-05"

# seconds each outside call waits, standing in for the network
WAIT = {"courier": 0.3, "payment": 0.4, "database": 0.05}


@dataclass
class Order:
    order_id: str
    customer_id: str
    item: str
    price_inr: int
    status: str            # shipped | delivered
    delivered_on: str | None


ORDERS = {o.order_id: o for o in [
    Order("58213", "40917", "wireless headphones", 2499, "delivered", "2026-10-18"),
    Order("61145", "40917", "screen protector", 399, "delivered", "2026-10-25"),
    Order("64410", "18843", "running shoes", 3199, "shipped", None),
    Order("64412", "18843", "water bottle", 549, "shipped", None),
]}

# what the courier says live, when it answers
LIVE_TRACKING = {
    "64410": ("out for delivery, Andheri East", "2026-11-05"),
    "64412": ("Delhi air cargo hub", "2026-11-07"),
}

# the last scan the shop saved in its own database: older, but always there (the fallback)
LAST_SCAN = {
    "64410": ("Bhiwandi sorting hub", "2026-11-04"),
    "64412": ("Delhi air cargo hub", "2026-11-03"),
}


class ServiceError(Exception):
    """An outside service failed. `transient` means the same call might work if you try again."""

    def __init__(self, status: int, message: str, transient: bool):
        super().__init__(message)
        self.status, self.message, self.transient = status, message, transient


@dataclass
class World:
    """What actually happened, so the score is read off the world, not off the agent's reply."""
    faults: dict = field(default_factory=dict)          # {"courier": "busy" | "down", "payment": "lost_reply"}
    courier_calls: int = 0
    payment_calls: int = 0
    refunds_paid: list[tuple[str, int]] = field(default_factory=list)   # (order_id, amount)
    payment_keys: dict = field(default_factory=dict)    # idempotency key -> saved reply
    attempts: list[str] = field(default_factory=list)   # every try at an outside call, with its outcome
    model_calls: list[str] = field(default_factory=list)
    reply: str = ""

    # ---- the courier -------------------------------------------------------------------------
    def courier_track(self, order_id: str) -> tuple[str, str]:
        self.courier_calls += 1
        time.sleep(WAIT["courier"])
        mode = self.faults.get("courier")
        if mode == "down" or (mode == "busy" and self.courier_calls <= 2):
            raise ServiceError(503, "The courier's tracking service is busy.", transient=True)
        return LIVE_TRACKING[order_id]

    # ---- the payment provider ----------------------------------------------------------------
    def payment_refund(self, order_id: str, amount: int, key: str | None) -> dict:
        """Pays a refund. With an idempotency key, a repeat of the same key returns the saved reply
        instead of paying again, which is how real payment APIs make a retry safe."""
        self.payment_calls += 1
        time.sleep(WAIT["payment"])
        if key is not None and key in self.payment_keys:
            return self.payment_keys[key]
        self.refunds_paid.append((order_id, amount))
        reply = {"refund_id": f"rf_{len(self.refunds_paid):03d}", "order_id": order_id, "amount_inr": amount}
        if key is not None:
            self.payment_keys[key] = reply
        if self.faults.get("payment") == "lost_reply" and self.payment_calls == 1:
            raise ServiceError(504, "The payment provider did not answer in time.", transient=True)
        return reply
