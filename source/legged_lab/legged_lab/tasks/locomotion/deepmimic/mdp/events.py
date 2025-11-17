from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm
    
def reset_from_ref(
    env: ManagerBasedAnimationEnv, 
    env_ids: torch.Tensor, 
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    root_states = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    
    root_states[env_ids, :3] = animation_term.get_root_pos_w(env_ids)[:, 0, :]
    root_states[env_ids, 3:7] = animation_term.get_root_quat(env_ids)[:, 0, :]
    root_states[env_ids, 7:10] = animation_term.get_root_vel_w(env_ids)[:, 0, :]
    root_states[env_ids, 10:13] = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]
    joint_pos[env_ids, :] = animation_term.get_dof_pos(env_ids)[:, 0, :]
    joint_vel[env_ids, :] = animation_term.get_dof_vel(env_ids)[:, 0, :]
    
    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    
    
    