"""Lesson 16, offline tests: no model, no network. Run with `python -m pytest` or `python test_errors.py`.

The waits are shortened so the tests are quick.
"""
import handler
import shop
from handler import raw, retry_all, smart
from shop import World

shop.WAIT.update({"courier": 0.0, "payment": 0.0, "database": 0.0})
handler.BASE_WAIT = 0.001

TRACK_64410 = ("shipping_track_package", {"order_id": "64410"})
TRACK_DELIVERED = ("shipping_track_package", {"order_id": "58213"})
TRACK_64412 = ("shipping_track_package", {"order_id": "64412"})
REFUND = ("payments_refund", {"order_id": "58213"})


def test_raw_hands_a_busy_courier_straight_to_the_model():
    world = World(faults={"courier": "busy"})
    assert raw(world, *TRACK_64410) == {"error": "HTTP 503"}
    assert world.courier_calls == 1


def test_smart_retries_a_busy_courier_and_the_model_never_sees_it():
    world = World(faults={"courier": "busy"})
    result = smart(world, *TRACK_64410)
    assert result["live"] is True and "Andheri" in result["now_at"]
    assert world.courier_calls == 3


def test_retrying_a_wrong_call_gets_the_same_answer_three_times():
    world = World()
    assert retry_all(world, *TRACK_DELIVERED) == {"error": "HTTP 404"}
    assert world.courier_calls == 3


def test_smart_sends_a_wrong_call_back_once_with_what_is_true():
    world = World()
    result = smart(world, *TRACK_DELIVERED)
    assert "delivered on 2026-10-18" in result["error"]
    assert world.courier_calls == 1


def test_smart_falls_back_to_the_last_scan_when_the_courier_is_down():
    world = World(faults={"courier": "down"})
    result = smart(world, *TRACK_64412)
    assert result["live"] is False and result["last_scan_at"] == "Delhi air cargo hub"
    assert world.courier_calls == 3


def test_retrying_a_refund_without_a_key_pays_twice():
    world = World(faults={"payment": "lost_reply"})
    result = retry_all(world, *REFUND)
    assert result["ok"] is True
    assert len(world.refunds_paid) == 2


def test_smart_retries_the_refund_with_a_key_and_pays_once():
    world = World(faults={"payment": "lost_reply"})
    result = smart(world, *REFUND)
    assert result["ok"] is True
    assert len(world.refunds_paid) == 1 and world.payment_calls == 2


def test_the_key_also_stops_a_second_refund_the_model_asks_for():
    world = World()
    smart(world, *REFUND)
    smart(world, *REFUND)
    assert len(world.refunds_paid) == 1


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
