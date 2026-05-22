#!/usr/bin/env python3
from __future__ import annotations

import os

import rospy


class OperatorPipelineNode:
    def __init__(self) -> None:
        self._model_path = rospy.get_param("~model_path", "")
        self._config_namespace = rospy.get_param(
            "~config_namespace",
            "/sage_ma_exploration/operator_pipeline",
        )

        self._preprocess_cfg = rospy.get_param(
            f"{self._config_namespace}/preprocess",
            {},
        )
        self._inference_cfg = rospy.get_param(
            f"{self._config_namespace}/inference",
            {},
        )
        self._postprocess_cfg = rospy.get_param(
            f"{self._config_namespace}/postprocess",
            {},
        )
        self._safety_cfg = rospy.get_param(
            f"{self._config_namespace}/safety_filter",
            {},
        )

    def validate(self) -> None:
        if self._model_path:
            if not os.path.exists(self._model_path):
                rospy.logwarn(
                    "MODEL_PATH is set but the file is not accessible: %s",
                    self._model_path,
                )
        else:
            rospy.loginfo("MODEL_PATH is empty; running with external model disabled.")

        rospy.loginfo(
            "Operator pipeline loaded: preprocess=%s inference=%s postprocess=%s safety=%s",
            sorted(self._preprocess_cfg.keys()),
            sorted(self._inference_cfg.keys()),
            sorted(self._postprocess_cfg.keys()),
            sorted(self._safety_cfg.keys()),
        )

    def run(self) -> None:
        self.validate()
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("operator_pipeline", anonymous=False)
    OperatorPipelineNode().run()

