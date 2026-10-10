import json

from needle_timemachine.trace_analysis import analyze_trace, to_dot


def test_analysis_summarizes_ops_layers_and_tensor_metadata():
    payload = {
        "format": "needle-timemachine.trace/v1",
        "checkpoint": "demo.pkl",
        "events": [
            {"step": 0, "op": "layer_output", "layer": 0,
             "tensors": {"output": {"shape": [1, 4, 8], "dtype": "float32", "nbytes": 128}},
             "metadata": {"runtime": True}},
            {"step": 1, "op": "attention", "layer": 0,
             "tensors": {"q": {"shape": [1, 4, 8], "dtype": "float32", "nbytes": 128}},
             "metadata": {"runtime": True}},
            {"step": 2, "op": "layer_output", "layer": 1, "tensors": {}},
        ],
    }
    report = analyze_trace(payload)
    summary = report["summary"]
    assert summary["event_count"] == 3
    assert summary["runtime_event_count"] == 2
    assert summary["layer_count"] == 2
    assert summary["operation_counts"] == {"attention": 1, "layer_output": 2}
    assert summary["tensor_record_count"] == 2
    assert summary["known_tensor_bytes_sum"] == 256
    assert summary["known_tensor_bytes_peak_per_event"] == 128
    assert report["tensors"][0]["shape"] == [1, 4, 8]


def test_dot_uses_only_explicit_dependencies():
    payload = {"events": [
        {"step": 0, "op": "qkv", "metadata": {}},
        {"step": 1, "op": "attention", "metadata": {"depends_on": [0]}},
        {"step": 2, "op": "output", "metadata": {}},
    ]}
    dot = to_dot(payload)
    assert '"0" -> "1"' in dot
    assert '"1" -> "2"' not in dot


def test_report_is_json_serializable():
    report = analyze_trace({"events": []})
    json.dumps(report)
    assert report["summary"]["event_count"] == 0
