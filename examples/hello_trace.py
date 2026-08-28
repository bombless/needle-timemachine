"""Run without JAX: this demonstrates the timeline/replay API."""

import numpy as np

from needle_timemachine import ReplayCursor, Tracer


tracer = Tracer(snapshot_every=2)
with tracer.session():
    x = np.zeros((1, 8, 512), dtype=np.float32)
    tracer.emit("embedding", tensors={"output": x})
    for layer in range(3):
        x = x + 1
        tracer.emit(
            "block_output",
            layer=layer,
            name=f"layers.{layer}",
            tensors={"output": x},
            state={"hidden": x},
        )

print("events:")
for event in tracer.events:
    print(event.step, event.name, event.layer, event.snapshot_id)

cursor = ReplayCursor(tracer.events, tracer.snapshots)
cursor.seek_step(2)
print("current:", cursor.current)
print("snapshot:", cursor.snapshot())
