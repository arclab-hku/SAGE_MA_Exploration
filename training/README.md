# Training Components

This directory contains the policy-training and simulation components used by SAGE.

`gnn_marl_planner/` contains the graph neural network actor-critic modules, action distributions, replay buffers, graph construction utilities, inference adapters, and training configuration parser.

`sim/` contains the grid/graph exploration environment, frontier extraction, communication/merge logic, baseline policies, metrics, and unit tests.

Checkpoints, trained weights, TensorBoard logs, generated datasets, and machine-local runtime artifacts are excluded.
