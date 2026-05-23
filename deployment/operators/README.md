# Operator Pipeline

The online policy path is organized as:

```text
map and graph state
  -> feature preprocessing
  -> utility inference adapter
  -> candidate postprocessing
  -> safety and feasibility filter
  -> selected exploration target
```

The corresponding implementation components are exposed in:

- `training/gnn_marl_planner/occupancy_to_graph.py`
- `training/gnn_marl_planner/model_inference_mixin.py`
- `training/gnn_marl_planner/goal_dispatch_mixin.py`
- `training/gnn_marl_planner/viewpoint_ops_mixin.py`
