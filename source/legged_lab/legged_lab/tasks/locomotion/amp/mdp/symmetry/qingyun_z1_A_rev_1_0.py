"""Functions to specify the symmetry in the observation and action space for Qingyun Z1-A rev 1.0 21dof."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    joint_num = 21
    key_body_num = 6

    history_len = 5
    ang_vel_dim = 3
    rot_tan_norm_dim = 6
    vel_cmd_dim = 3
    joint_pos_dim = joint_num
    joint_vel_dim = joint_num
    last_actions_dim = joint_num
    key_body_pos_dim = key_body_num * 3

    end_idx = 0
    for _ in range(history_len):
        start_idx = end_idx
        end_idx = start_idx + ang_vel_dim
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)
    for _ in range(history_len):
        start_idx = end_idx
        end_idx = start_idx + rot_tan_norm_dim
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1, 1, -1, 1], device=device)
    for _ in range(history_len):
        start_idx = end_idx
        end_idx = start_idx + vel_cmd_dim
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)
    for _ in range(history_len):
        start_idx = end_idx
        end_idx = start_idx + joint_pos_dim
        obs[:, start_idx:end_idx] = _switch_joints_left_right(obs[:, start_idx:end_idx])
    for _ in range(history_len):
        start_idx = end_idx
        end_idx = start_idx + joint_vel_dim
        obs[:, start_idx:end_idx] = _switch_joints_left_right(obs[:, start_idx:end_idx])
    for _ in range(history_len):
        start_idx = end_idx
        end_idx = start_idx + last_actions_dim
        obs[:, start_idx:end_idx] = _switch_joints_left_right(obs[:, start_idx:end_idx])
    for _ in range(history_len):
        start_idx = end_idx
        end_idx = start_idx + key_body_pos_dim
        obs[:, start_idx:end_idx] = _switch_key_body_pos_left_right(obs[:, start_idx:end_idx])

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    actions = actions.clone()
    actions[:] = _switch_joints_left_right(actions[:])
    return actions

"""
Lab joint names:
 0 - lw_shoulder_pitch
 1 - rw_shoulder_pitch
 2 - w_waist_yaw
 3 - lw_arm_roll
 4 - rw_arm_roll
 5 - lw_hip_pitch
 6 - rw_hip_pitch
 7 - lw_arm_yaw
 8 - rw_arm_yaw
 9 - lw_hip_roll
 10 - rw_hip_roll
 11 - lw_elbow_pitch
 12 - rw_elbow_pitch
 13 - lw_hip_yaw
 14 - rw_hip_yaw
 15 - lw_knee_pitch
 16 - rw_knee_pitch
 17 - lw_foot_pitch
 18 - rw_foot_pitch
 19 - lw_foot_roll
 20 - rw_foot_roll

左侧关节 → 右侧关节 交换:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 0 lw_shoulder_pitch   ↔ 1 rw_shoulder_pitch
 3 lw_arm_roll         ↔ 4 rw_arm_roll
 5 lw_hip_pitch        ↔ 6 rw_hip_pitch
 7 lw_arm_yaw          ↔ 8 rw_arm_yaw
 9 lw_hip_roll         ↔ 10 rw_hip_roll
 11 lw_elbow_pitch     ↔ 12 rw_elbow_pitch
 13 lw_hip_yaw         ↔ 14 rw_hip_yaw
 15 lw_knee_pitch      ↔ 16 rw_knee_pitch
 17 lw_foot_pitch      ↔ 18 rw_foot_pitch
 19 lw_foot_roll       ↔ 20 rw_foot_roll

中间关节（不交换）:
 2 w_waist_yaw
"""
def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Joint order follows scripts/tools/retarget/config/qingyun_z1_A_rev_1_0.yaml."""
    joint_data_switched = torch.zeros_like(joint_data)

    left_indices = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    right_indices = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    joint_data_switched[..., 2] = -joint_data[..., 2]
    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    roll_indices = [3, 4, 9, 10, 19, 20, 15, 16]
    yaw_indices = [7, 8, 13, 14]

    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0
    return joint_data_switched


def _switch_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    key_body_pos_switched = key_body_pos.clone()
    num_key_bodies = key_body_pos.shape[-1] // 3

    # KEY_BODY_NAMES = [
    #     "lp_foot_roll_link",
    #     "rp_foot_roll_link",
    #     "lp_elbow_pitch_link",
    #     "rp_elbow_pitch_link",
    #     "lp_arm_roll_link",
    #     "rp_arm_roll_link",
    # ]

    for i in range(num_key_bodies // 2):
        left_idx = i * 2
        right_idx = i * 2 + 1
        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
            ..., right_idx * 3 : right_idx * 3 + 3
        ]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
            ..., left_idx * 3 : left_idx * 3 + 3
        ]
        key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

    return key_body_pos_switched
