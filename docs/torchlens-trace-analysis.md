# TorchLens-inspired trace analysis

Needle Time Machine already records layer and optional operation-level runtime
events. This companion analysis command borrows the useful inspection pattern
from [the TorchLens experiment notebook](https://github.com/xuyuji9000/machine-learning-playground/blob/master/01_tasks/99_DONE/20261009-1_try-out-torchlens/main.ipynb):
a compact operation summary, tensor shape/dtype/byte metadata, and an exportable
graph for inspecting a forward pass.

## Run

After generating a trace as usual:

```bash
python -m needle_timemachine.trace_needle --trace-level op --output traces/run.json
python -m needle_timemachine.trace_analysis traces/run.json --report traces/analysis.json --dot traces/graph.dot
```

Omit `--report` to print the report to stdout. Render the DOT file with
Graphviz, for example `dot -Tsvg traces/graph.dot -o traces/graph.svg`.

The report includes event counts by operation, distinct layer IDs, runtime
event count, each recorded tensor's shape/dtype/byte count, and conservative
memory summaries. Byte totals are metadata sums, not peak device-memory
measurements; aliases and repeated references can be counted more than once.

**Important graph limitation:** the current trace schema does not always retain
tensor-level parent IDs. The DOT exporter therefore draws edges only when an
event explicitly provides `metadata.depends_on` or `metadata.inputs`; it does
not invent data-flow edges from event order. FLOPs and parameter counts are not
estimated unless the trace itself records them. This keeps the report honest
while creating a foundation for richer module-level accounting and hierarchical
layer/operation browsing.
