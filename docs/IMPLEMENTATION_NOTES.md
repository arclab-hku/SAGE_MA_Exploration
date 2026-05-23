# Implementation Notes

The repository exposes the custom components needed to inspect the SAGE method:

- Graph/map state representation for frontiers, candidate viewpoints, robot nodes, and navigation edges.
- Operator pipeline for converting graph state into utility scores.
- Multi-agent target selection with distance and reservation penalties.
- Versioned merge helpers for exchanging graph updates between agents.
- GNN actor-critic and graph-policy components used by the training code.
- Grid/graph simulation components used for policy development and testing.
- ROS messages, launch files, and deployment parameters for the online system.

Large experiment artifacts, datasets, and trained weights are excluded from the repository.
