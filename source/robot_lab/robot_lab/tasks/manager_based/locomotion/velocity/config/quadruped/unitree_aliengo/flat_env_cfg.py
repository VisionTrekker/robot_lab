# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.terrains import FLAT_TERRAINS_CFG
from .rough_env_cfg import UnitreeAliengoRoughEnvCfg


@configclass
class UnitreeAliengoFlatEnvCfg(UnitreeAliengoRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # change terrain to flat
        self.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG

        # ------------------------------Observations------------------------------
        self.observations.policy.height_scan = None

        # ------------------------------Rewards------------------------------
        self.rewards.base_height_l2.weight = 0.5
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeAliengoFlatEnvCfg":
            self.disable_zero_weight_rewards()
