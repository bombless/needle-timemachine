"""TorchLens-inspired structural analysis for saved Needle Time Machine traces.

This module deliberately reports only facts present in the trace. It does not
pretend that chronological adjacency is a tensor dependency: graph edges are
drawn only when an event explicitly declares input/dependency event IDs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _tensor_records(event: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    tensors = event.get("tensors", {})
    if not isinstance(tensors, dict):
        return records
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            continue
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        nbytes = tensor.get("nbytes")
        records.append({
            "name": str(name),
            "shape": list(shape) if isinstance(shape, (list, tuple)) else shape,
            "dtype": str(dtype) if dtype is not None else None,
            "nbytes": int(nbytes) if isinstance(nbytes, (int, float)) else None,
        })
    return records


def analyze_trace(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a compact model/operation/activation report for a trace JSON object."""
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError("trace 'events' must be a list")

    op_counts: Counter[str] = Counter()
    layer_ids: set[Any] = set()
    runtime_count = 0
    tensors_seen = 0
    tensor_bytes_by_event: list[int] = []
    tensor_rows = []
    event_ids: set[str] = set()

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        op = str(event.get("op", "unknown"))
        op_counts[op] += 1
        if event.get("layer") is not None:
            layer_ids.add(event["layer"])
        if (event.get("metadata") or {}).get("runtime"):
            runtime_count += 1
        event_id = str(event.get("id", event.get("step", index)))
        event_ids.add(event_id)
        event_total = 0
        for tensor in _tensor_records(event):
            tensors_seen += 1
            if tensor["nbytes"] is not None:
                event_total += tensor["nbytes"]
            tensor_rows.append({
                "event_id": event_id,
                "step": event.get("step", index),
                "op": op,
                "event_name": event.get("name"),
                **tensor,
            })
        tensor_bytes_by_event.append(event_total)

    return {
        "format": "needle-timemachine.analysis/v1",
        "source_format": payload.get("format"),
        "checkpoint": payload.get("checkpoint"),
        "prompt": payload.get("prompt"),
        "summary": {
            "event_count": len(events),
            "runtime_event_count": runtime_count,
            "layer_count": len(layer_ids),
            "operation_counts": dict(sorted(op_counts.items())),
            "tensor_record_count": tensors_seen,
            "known_tensor_bytes_sum": sum(row["nbytes"] or 0 for row in tensor_rows),
            "known_tensor_bytes_peak_per_event": max(tensor_bytes_by_event, default=0),
            "memory_note": "Byte totals sum recorded tensor metadata; shared/aliased buffers may be counted more than once.",
        },
        "tensors": tensor_rows,
    }


def to_dot(payload: dict[str, Any]) -> str:
    """Export a conservative event graph in Graphviz DOT format.

    Only explicit metadata keys 'depends_on' or 'inputs' create edges. Values
    may be event IDs or step numbers. Events without dependency metadata remain
    visible as isolated nodes rather than receiving invented edges.
    """
    events = payload.get("events", [])
    lines = ["digraph NeedleTrace {", '  rankdir="LR";', '  node [shape="box", fontname="monospace"];']
    lookup: dict[str, str] = {}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", event.get("step", index)))
        lookup[event_id] = event_id
        label = f"{event.get('step', index)}: {event.get('op', 'unknown')}"
        if event.get("name"):
            label += "\\n" + str(event["name"])
        label = label.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  "{event_id}" [label="{label}"];')
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        target = str(event.get("id", event.get("step", index)))
        metadata = event.get("metadata") or {}
        deps = metadata.get("depends_on", metadata.get("inputs", []))
        if not isinstance(deps, list):
            deps = [deps]
        for dep in deps:
            source = str(dep)
            if source in lookup and source != target:
                lines.append(f'  "{source}" -> "{target}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a Needle Time Machine trace (TorchLens-inspired).")
    parser.add_argument("trace", type=Path, help="Trace JSON produced by trace_needle")
    parser.add_argument("--report", type=Path, help="Write analysis JSON (defaults to stdout)")
    parser.add_argument("--dot", type=Path, help="Optionally export the explicit dependency graph as Graphviz DOT")
    args = parser.parse_args(argv)
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    report = analyze_trace(payload)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        args.dot.write_text(to_dot(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
