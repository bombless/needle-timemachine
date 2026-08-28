# Needle Time Machine

A step-by-step, inspectable execution tracer and time-machine UI for the Cactus Compute Needle model.

## Project direction

The first milestone is a Python/JAX instrumentation layer around Needle's model implementation. It should capture execution events and selected tensor snapshots without changing model semantics, then expose a replayable timeline for a future visual debugger.

## Initial architecture

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
        +----> timeline / replay API
        |
        v
    visualization UI
```

## Milestones

- [ ] Bootstrap project and development instructions
- [ ] Vendor or reference the upstream Needle source cleanly
- [ ] Trace Transformer block boundaries
- [ ] Trace attention and MLP sub-operations
- [ ] Add tensor metadata and optional snapshots
- [ ] Implement pause / step / seek / replay semantics
- [ ] Build a minimal timeline UI
- [ ] Validate numerical equivalence with uninstrumented Needle
- [ ] Investigate JIT/native-runtime tracing only after the Python prototype works

## Design principle

Do not start from the Hugging Face native executable. Start from the readable Python/JAX implementation so execution boundaries are observable and replay can be implemented as state snapshots rather than reverse execution.
