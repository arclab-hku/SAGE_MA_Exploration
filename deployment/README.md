# Deployment

This folder contains the environment files, launch wrappers, and configuration templates for online SAGE deployment.

## Layout

```text
deployment/
├── config/
│   ├── online_exploration.yaml
│   ├── operator_pipeline.yaml
│   └── scenarios/
├── launch/
│   └── online_exploration.launch
├── operators/
│   └── README.md
├── requirements.txt
├── env.example
└── scripts/
    ├── build_ros1_workspace.sh
    └── run_online_exploration.sh
```

## Setup

```bash
python3 -m pip install -r deployment/requirements.txt
deployment/scripts/build_ros1_workspace.sh
```

## Run

```bash
cp deployment/env.example .env
deployment/scripts/run_online_exploration.sh
```

The environment file controls the ROS workspace path, scenario configuration, planner limits, and optional learned-utility path.
