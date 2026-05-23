# SAGE_MA_Exploration

This repository provides the SAGE implementation corresponding to the revised manuscript.

SAGE is a ROS-based multi-agent exploration system organized around shared map/graph state, operator-based utility estimation, and decentralized target selection. The repository contains the core custom code, ROS interfaces, deployment configuration, and run scripts used to assess the method structure.

## Repository Structure

```text
SAGE_MA_Exploration/
├── deployment/                   # Environment, launch, and deployment scripts
├── docs/                         # Additional implementation notes
├── training/
│   ├── gnn_marl_planner/         # GNN policy, actor-critic, buffers, inference utilities
│   └── sim/                      # Grid/graph exploration simulation environment
├── ros1_ws/src/sage_ma_exploration/
│   ├── launch/                   # ROS launch files
│   ├── config/                   # ROS and planner parameters
│   ├── msg/                      # Custom ROS messages
│   └── rviz/                     # Visualization configuration
├── configs/                      # Scenario and experiment configuration
├── scripts/                      # Evaluation and visualization entry points
├── assets/                       # Documentation assets
└── tests/                        # Tests for core modules
```

## Environment

The reference environment is:

- Ubuntu 20.04
- ROS Noetic
- Python 3.8+
- `numpy`, `PyYAML`
- Catkin workspace under `ros1_ws`

Install the Python dependencies with:

```bash
python3 -m pip install -r deployment/requirements.txt
```

Build the ROS workspace with:

```bash
deployment/scripts/build_ros1_workspace.sh
```

## Online Deployment Entry Point

The deployment entry point is:

```bash
deployment/scripts/run_online_exploration.sh
```

Runtime parameters are configured through:

- `deployment/env.example`
- `deployment/config/online_exploration.yaml`
- `deployment/config/operator_pipeline.yaml`
- `deployment/config/scenarios/`

The repository does not include large runtime artifacts such as rosbags, datasets, checkpoints, or trained weight files.
