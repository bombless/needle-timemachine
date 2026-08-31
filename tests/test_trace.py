import base64

import numpy as np

from needle_timemachine import ReplayCursor, SnapshotStore, Tracer
from needle_timemachine.needle_adapter import NeedleAdapter


def test_trace_records_tensor_metadata_and_step():
    tracer = Tracer()
    value = np.ones((2, 3), dtype=np.float32)
    tracer.emit("layer", layer=0, name="layers.0", tensors={"output": value})
    event = tracer.events[0]
    assert tracer.step == 1
    assert event.layer == 0
    assert event.tensors["output"].shape == (2, 3)
    assert event.tensors["output"].dtype == "float32"
    assert event.tensors["output"].nbytes == 24


def test_trace_can_capture_browser_replayable_values():
    tracer = Tracer()
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tracer.emit("layer", values={"input": value})
    payload = tracer.events[0].values["input"]
    assert payload.shape == (2, 2)
    assert payload.encoding == "base64-f32-le"
    assert np.frombuffer(base64.b64decode(payload.data), dtype="<f4").tolist() == [1.0, 2.0, 3.0, 4.0]
    assert payload.sum == 10.0
    assert payload.sum_squares == 30.0


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


def test_needle_adapter_emits_layer_input_and_output():
    class FakeNeedle:
        def hidden_states(self, tokens, **kwargs):
            return np.arange(1 * 2 * 4 * 3, dtype=np.float32).reshape(1, 2, 4, 3)

    tracer = Tracer(summarize_tensors=False)
    adapter = NeedleAdapter(FakeNeedle(), tracer)
    result = adapter.hidden_states(np.array([[1, 2]], dtype=np.int32))
    assert result.shape == (1, 2, 4, 3)
    layer_events = [e for e in tracer.events if e.op == "layer_output"]
    assert [e.layer for e in layer_events] == [0, 1, 2]
    assert [e.metadata["state_index"] for e in layer_events] == [1, 2, 3]
    assert layer_events[0].tensors["input"].shape == (1, 2, 3)
    assert layer_events[0].tensors["output"].shape == (1, 2, 3)
    assert set(layer_events[0].values) == {"input", "output"}
    assert tracer.events[-1].metadata["layer_count"] == 3
