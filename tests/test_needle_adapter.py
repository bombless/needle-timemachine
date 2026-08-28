import numpy as np

from needle_timemachine import NeedleAdapter, Tracer


class FakeNeedle:
    def hidden_states(self, tokens, **kwargs):
        batch, seq = tokens.shape
        return np.arange(batch * seq * 3 * 2, dtype=np.float32).reshape(batch, seq, 3, 2)

    def __call__(self, tokens, **kwargs):
        return np.zeros((tokens.shape[0], tokens.shape[1], 8), dtype=np.float32)


def test_hidden_states_emits_embedding_and_each_transformer_layer():
    tracer = Tracer(summarize_tensors=False)
    adapter = NeedleAdapter(FakeNeedle(), tracer)
    tokens = np.array([[1, 2]], dtype=np.int32)
    cells = adapter.hidden_states(tokens)
    assert cells.shape == (1, 2, 3, 2)
    layer_events = [event for event in tracer.events if event.op == "layer_output"]
    assert [event.layer for event in layer_events] == [0, 1]
    assert layer_events[0].metadata["state_index"] == 1
    assert layer_events[1].metadata["state_index"] == 2
    assert tracer.events[1].op == "embedding_output"
    assert tracer.events[-1].metadata["layer_count"] == 2


def test_none_trace_level_skips_layer_events():
    tracer = Tracer(summarize_tensors=False)
    adapter = NeedleAdapter(FakeNeedle(), tracer, trace_level="none")
    adapter.hidden_states(np.array([[1, 2]], dtype=np.int32))
    assert not any(event.op == "layer_output" for event in tracer.events)
    assert tracer.events[-1].metadata["layer_trace"] is False


def test_record_hidden_states_does_not_recompute_model():
    class CountingNeedle(FakeNeedle):
        def __init__(self):
            self.calls = 0

        def hidden_states(self, tokens, **kwargs):
            self.calls += 1
            return super().hidden_states(tokens, **kwargs)

    model = CountingNeedle()
    tracer = Tracer(summarize_tensors=False)
    adapter = NeedleAdapter(model, tracer)
    tokens = np.array([[1, 2]], dtype=np.int32)
    cells = model.hidden_states(tokens)
    adapter.record_hidden_states(tokens, cells)
    assert model.calls == 1
    assert [e.layer for e in tracer.events if e.op == "layer_output"] == [0, 1]
