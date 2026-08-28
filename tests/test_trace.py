import numpy as np

from needle_timemachine import ReplayCursor, SnapshotStore, Tracer


def test_trace_records_tensor_metadata_and_step():
    tracer = Tracer()
    value = np.ones((2, 3), dtype=np.float32)
    tracer.emit("layer", layer=0, name="layers.0", tensors={"output": value})

    assert tracer.step == 1
    assert len(tracer.events) == 1
    event = tracer.events[0]
    assert event.layer == 0
    assert event.tensors["output"].shape == (2, 3)
    assert event.tensors["output"].dtype == "float32"
    assert event.tensors["output"].nbytes == 24


def test_snapshot_is_a_copy_and_replay_can_seek():
    store = SnapshotStore()
    value = np.array([1, 2, 3], dtype=np.float32)
    sid = store.put(1, {"x": value})
    value[:] = 9
    assert np.array_equal(store.get(sid).state["x"], [1, 2, 3])

    tracer = Tracer(snapshot_store=store)
    tracer.emit("a", layer=0)
    tracer.emit("b", layer=1)
    tracer.emit("c", layer=2)
    cursor = ReplayCursor(tracer.events, store)
    assert cursor.seek_step(2).op == "b"
    assert cursor.backward().op == "a"
    assert cursor.forward().op == "b"
