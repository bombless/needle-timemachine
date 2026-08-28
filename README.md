# Needle Time Machine

A step-by-step, inspectable execution tracer and time-machine UI for the Cactus Compute Needle model.

## Project direction

The project uses a Python/JAX instrumentation layer around Needle's readable model implementation. It captures execution events and selected tensor metadata without changing model semantics, then exposes a replayable timeline for a visual debugger.

## Current prototype

The `feat/timeline-ui` branch adds a zero-dependency browser timeline on top of the existing trace format:

```text
Needle model (Python/JAX)
        |
        v
  instrumentation
        |
        v
   TraceEvent stream
        |
        +----> snapshot store
        |
        +----> replay cursor
        |
        v
   local timeline UI
```

The UI supports **play/pause, speed control, previous/next, seek, layer markers, tensor metadata and event metadata**. It replays an already captured trace, so going backward is deterministic and does not require reverse execution.

### Run it

Generate a trace as before, or let the CLI serve it directly:

```bash
python -m needle_timemachine.trace_needle \
  --checkpoint /path/to/checkpoint \
  --needle-source /path/to/needle \
  --prompt "hello world" \
  --output traces/run.json \
  --serve
```

To serve an existing trace:

```bash
python -m needle_timemachine.ui traces/run.json
```

Then open `http://127.0.0.1:8765/`.

## Milestones

- [x] Bootstrap project and development instructions
- [x] Vendor or reference the upstream Needle source cleanly
- [x] Trace Transformer block boundaries
- [x] Add tensor metadata and optional snapshots
- [x] Implement pause / step / seek / replay semantics
- [x] Build a minimal timeline UI
- [ ] Trace attention and MLP sub-operations
- [ ] Persist snapshots into the trace file for richer tensor inspection
- [ ] Validate numerical equivalence with uninstrumented Needle
- [ ] Investigate JIT/native-runtime tracing only after the Python prototype works

## Design principle

Do not start from the Hugging Face native executable. Start from the readable Python/JAX implementation so execution boundaries are observable and replay can be implemented as state snapshots rather than reverse execution.
