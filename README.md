# SAGE_MA_Exploration

This repository is a public placeholder for the SAGE multi-agent exploration project during the review process.

The complete code release is being prepared and will be made available in this repository. The release will include the ROS simulation package, learned-policy interfaces, planning components, evaluation scripts, configuration files, and instructions needed to reproduce the reported experiments.

At this stage, the repository is intentionally kept as a lightweight shell to provide reviewers with a stable project URL while the implementation is being cleaned, documented, and packaged for public use.

## Planned Repository Structure

```text
SAGE_MA_Exploration/
├── deployment/                   # Clean online deployment templates
├── docs/                         # Reviewer notes and release documentation
├── ros1_ws/src/sage_ma_exploration/
│   ├── launch/                   # ROS launch files
│   ├── config/                   # ROS and planner parameters
│   ├── msg/                      # Custom ROS messages, if needed
│   ├── rviz/                     # Visualization configurations
│   └── src/
│       ├── mapping/              # Local map and occupancy processing
│       ├── graph/                # Global graph construction and update logic
│       ├── policy/               # Learned utility and decision modules
│       ├── planning/             # Frontier, path, and task planning modules
│       └── communication/        # Multi-agent information sharing modules
├── configs/
│   ├── scenarios/                # Environment and robot setup files
│   └── experiments/              # Evaluation and ablation configurations
├── scripts/
│   ├── training/                 # Training entry points
│   ├── evaluation/               # Batch evaluation and metric scripts
│   └── visualization/            # Plotting and visualization utilities
├── assets/figures/               # Figures and lightweight visual assets
└── tests/                        # Unit and integration tests
```

## Release Status

- Public repository shell: available.
- Clean deployment scaffold without trained weights: available.
- Full source code and reproduction instructions: preparing for release.
- Supplementary scripts, configuration files, and experiment assets: preparing for release.

For questions during the review period, please refer to the accompanying manuscript and supplementary material.
