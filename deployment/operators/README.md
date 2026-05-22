# Operator Pipeline

The online policy path is organized as a small operator pipeline:

```text
map and graph state
  -> feature preprocessing
  -> utility inference adapter
  -> candidate postprocessing
  -> safety and feasibility filter
  -> selected exploration target
```

The public repository includes the interface, launch hook, and configuration template for this pipeline. Trained weights and hardware-specific runtime assets are intentionally excluded.

External weights can be provided at runtime through `MODEL_PATH` after the full release is available.

