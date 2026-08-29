# Needle Time Machine

A step-by-step, inspectable execution tracer and time-machine UI for the Cactus Compute Needle model.

## Project direction

The project uses a Python/JAX instrumentation layer around Needle's readable model implementation. It captures execution events and selected tensor metadata, then exposes a replayable timeline for a visual debugger.

The upstream Needle checkout is kept separate. For a local Windows setup such as `d:\needle2\needle`, Time Machine can install a small, reversible instrumentation patch into that checkout instead of copying or forking Needle sources.

## Current prototype

The `feat/timeline-ui` branch contains two trace modes:

- `layer`: records the embedding and every Transformer layer output.
- `op`: records runtime values inside every layer: norm input/output, Q/K/V, attention output/projection, residual, Hadamard MLP stages, and—when Needle is running its non-flash attention path—attention scores and softmax probabilities.

The operation-level bridge uses `jax.debug.callback`, which is designed to observe runtime values inside staged JAX programs and is compatible with `scan`, `jit`, `vmap` and `grad`. citeturn2search1turn2search2

```text
Needle model (Python/JAX)
        |
        +---- layer scan
        |       |
        |       +---- norm
        |       +---- Q / K / V
        |       +---- attention
        |       +---- residual
        |       +---- Hadamard MLP
        |
        v
   runtime callbacks
        |
        v
   TraceEvent stream
        |
        +----> snapshot store
        +----> replay cursor
        |
        v
   local timeline UI
```

Needle itself is a 27-layer encoder/decoder-style Simple Attention Network in the current source tree, with GQA, RoPE, gated residuals, Engram sites and a Hadamard MLP. The exact current implementation is in `needle/model/architecture.py`; the model's public Python API is also documented upstream. citeturn0search1turn0search8

## Windows: use your local Needle checkout

Given:

```text
d:\needle2\needle\
```

run from the Time Machine repository:

```bat
py tools\patch_needle_trace.py d:\needle2\needle
```

This changes only:

```text
d:\needle2\needle\needle\model\architecture.py
```

and creates:

```text
d:\needle2\needle\needle\model\architecture.py.timemachine.bak
```

The patch is idempotent. It adds `set_timemachine_hook()` plus runtime callbacks; the normal Needle computation is otherwise left intact. To remove the instrumentation, restore the `.timemachine.bak` file.

## Generate a trace

Layer-level trace:

```bat
python -m needle_timemachine.trace_needle ^
  --checkpoint d:\path\to\needle.pkl ^
  --needle-source d:\needle2\needle ^
  --prompt "hello world" ^
  --output traces\run.json ^
  --trace-level layer ^
  --serve
```

Operation-level trace:

```bat
python -m needle_timemachine.trace_needle ^
  --checkpoint d:\path\to\needle.pkl ^
  --needle-source d:\needle2\needle ^
  --prompt "hello world" ^
  --output traces\run.json ^
  --trace-level op ^
  --serve
```

Then open `http://127.0.0.1:8765/`.

## Tool-calling evaluation workbench

`tool_eval` is a separate, zero-dependency browser module that starts its own OpenAI-compatible `/v1/chat/completions` endpoint. The browser and model endpoint share one port; no second inference server URL is needed in the page. It includes templates for exact selection, similar-tool disambiguation, parallel calls, missing-argument clarification, prompt-injection resistance, strict JSON types, and no-tool answers.

Start it with a real Needle 2 checkpoint:

```bat
py -3.11 -m needle_timemachine.tool_eval ^
  --checkpoint d:\path\to\needle2.pkl ^
  --needle-source d:\needle2\needle ^
  --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787/` and run an individual case or the full set. The built-in endpoint is `POST /v1/chat/completions`; the evaluator calls it internally through `POST /api/run`. The evaluator scores tool choice, required argument values, and cases where a tool call must not happen. The prompt and tools JSON remain editable so the templates can be adapted to the model.

Operation-level tracing is intentionally a debugger mode: runtime callbacks can perturb compilation/fusion and add substantial overhead. JAX explicitly warns that debugging callbacks can change the compiled computation, so production performance should always be measured with tracing disabled. citeturn1search7

## Current visual debugger

The browser timeline supports **play/pause, speed control, previous/next, seek, layer markers, tensor metadata and event metadata**. It replays an already captured trace, so going backward is deterministic and does not require reverse execution.

The next step is to make the timeline hierarchical:

```text
Layer 7
 ├─ RMSNorm
 ├─ Q / K / V
 ├─ Attention scores
 ├─ Softmax
 ├─ Attention output
 ├─ Output projection
 ├─ Attention residual
 ├─ Hadamard #1
 ├─ SiLU
 ├─ Hadamard #2
 └─ Layer output
```

For flash attention, scores/softmax remain inside the fused kernel and therefore are not exposed by this source-level hook; the operation trace still exposes Q/K/V and the fused attention output. A non-flash debug configuration can expose the intermediate score and softmax tensors.

## Milestones

- [x] Bootstrap project and development instructions
- [x] Reference the upstream Needle source cleanly
- [x] Trace Transformer block boundaries
- [x] Add tensor metadata and optional snapshots
- [x] Implement pause / step / seek / replay semantics
- [x] Build a minimal timeline UI
- [x] Add runtime operation hooks to local Needle
- [ ] Make the UI hierarchical by layer/operation
- [ ] Persist runtime tensor snapshots into the trace file
- [ ] Add live pause/resume while the JAX program is executing
- [ ] Validate numerical equivalence with uninstrumented Needle
- [ ] Bridge the same event model to `needle.exe` / `libneedle.a`

## Design principle

Do not start from the native executable. Start from the readable Python/JAX implementation so execution boundaries are observable and replay can be implemented as state snapshots rather than reverse execution. The native `needle.exe` and `libneedle.a` can become a later backend once the Python/JAX debugger semantics are stable.
