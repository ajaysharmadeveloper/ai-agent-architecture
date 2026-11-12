"""Lesson 15, offline tests: no model, no network. Run with `python -m pytest` or `python test_parallel.py`.

The waits are shortened so the tests are quick; the ratios are what matter.
"""
import shop
from executor import parallel, safe, sequential, timed
from shop import OLD_ADDRESS, World

shop.WAIT.update({"orders_get_order": 0.05, "shipping_track_package": 0.15,
                  "account_update_address": 0.08, "orders_ship_replacement": 0.1})

THREE_TRACKS = [("shipping_track_package", {"order_id": o}) for o in ("64410", "64411", "64412")]
MOVE_THEN_SHIP = [("account_update_address", {"customer_id": "40917", "address": "14 MG Road, Pune 411001"}),
                  ("orders_ship_replacement", {"order_id": "58213"})]


def test_three_reads_in_parallel_take_about_one_wait():
    seq, par = World(), World()
    timed(sequential, seq, THREE_TRACKS)
    timed(parallel, par, THREE_TRACKS)
    assert seq.tool_seconds > 0.4          # three waits of 0.15
    assert par.tool_seconds < 0.25         # one wait, the three overlap


def test_parallel_write_then_read_ships_to_the_old_address():
    world = World()
    parallel(world, MOVE_THEN_SHIP)
    assert world.shipped_to == [("58213", OLD_ADDRESS)]


def test_sequential_write_then_read_ships_to_the_new_address():
    world = World()
    sequential(world, MOVE_THEN_SHIP)
    assert world.shipped_to == [("58213", "14 MG Road, Pune 411001")]


def test_safe_runs_reads_together_and_writes_in_order():
    world = World()
    timed(safe, world, THREE_TRACKS + MOVE_THEN_SHIP)
    assert world.shipped_to == [("58213", "14 MG Road, Pune 411001")]
    assert world.tool_seconds < 0.15 * 2 + 0.08 + 0.1   # the three tracks overlapped


def test_safe_keeps_the_models_order():
    """safe does not reorder. If the model asks to ship first, the parcel goes to the old address."""
    world = World()
    safe(world, list(reversed(MOVE_THEN_SHIP)))
    assert world.shipped_to == [("58213", OLD_ADDRESS)]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
