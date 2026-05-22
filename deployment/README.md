# Deployment

This folder provides a clean online deployment scaffold for the multi-agent exploration system.

It contains ROS launch wrappers, parameter templates, and helper scripts. It does not contain trained weights, checkpoints, rosbags, logs, datasets, or large runtime artifacts.

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
├── scripts/
│   ├── build_ros1_workspace.sh
│   └── run_online_exploration.sh
└── env.example
```

## Basic Use

```bash
cp deployment/env.example .env
deployment/scripts/build_ros1_workspace.sh
deployment/scripts/run_online_exploration.sh
```

Set `MODEL_PATH` in the environment only when running with external trained weights stored outside this repository.
