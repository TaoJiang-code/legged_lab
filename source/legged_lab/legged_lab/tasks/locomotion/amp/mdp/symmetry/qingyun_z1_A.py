
"""Functions to specify the symmetry in the observation and action space for Qingyun Z1-A 19dof."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size:2*batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    joint_num = 19  # qingyun_z1_A 19dof
    key_body_num = 6

    HISTORY_LEN = 5
    ANG_VEL_DIM = 3
    ROT_TAN_NORM = 6
    VEL_CMD_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    KEY_BODY_POS_DIM = key_body_num * 3

    end_idx = 0
    # ang vel
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)
    # root rot tan norm
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ROT_TAN_NORM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1, 1, -1, 1], device=device)
    # velocity command
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_qingyun_z1_A_joints_left_right(obs[:, start_idx:end_idx])
    # joint vel
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_qingyun_z1_A_joints_left_right(obs[:, start_idx:end_idx])
    # last actions
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_qingyun_z1_A_joints_left_right(obs[:, start_idx:end_idx])
    # key body pos
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + KEY_BODY_POS_DIM
        obs[:, start_idx:end_idx] = _switch_qingyun_z1_A_key_body_pos_left_right(obs[:, start_idx:end_idx])

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_qingyun_z1_A_joints_left_right(actions[:])
    return actions


"""
Lab joint names (19 dof, alphabetical order assigned by IsaacLab):
 0 - lw_shoulder_pitch
 1 - rw_shoulder_pitch
 2 - w_waist_yaw
 3 - lw_arm_roll
 4 - rw_arm_roll
 5 - lrw_hip_pitch
 6 - rw_hip_pitch
 7 - lw_forearm_yaw
 8 - rw_forearm_yaw
 9 - lrw_hip_roll
10 - rw_hip_roll
11 - lrw_hip_yaw
12 - rw_hip_yaw
13 - lrw_knee_pitch
14 - rw_knee_pitch
15 - lrw_foot_pitch
16 - rw_foot_pitch
17 - lrw_foot_roll
18 - rw_foot_roll

左侧关节 → 右侧关节 交换:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 0 lw_shoulder_pitch  ↔  1 rw_shoulder_pitch
 3 lw_arm_roll        ↔  4 rw_arm_roll
 5 lrw_hip_pitch      ↔  6 rw_hip_pitch
 7 lw_forearm_yaw     ↔  8 rw_forearm_yaw
 9 lrw_hip_roll       ↔ 10 rw_hip_roll
11 lrw_hip_yaw        ↔ 12 rw_hip_yaw
13 lrw_knee_pitch     ↔ 14 rw_knee_pitch
15 lrw_foot_pitch     ↔ 16 rw_foot_pitch
17 lrw_foot_roll      ↔ 18 rw_foot_roll

中间关节（不交换，但waist_yaw需取反）:
 2 w_waist_yaw  (取反)

需要取反的关节（交换后）:
  roll类: lw_arm_roll(3), rw_arm_roll(4), lrw_hip_roll(9), rw_hip_roll(10), lrw_foot_roll(17), rw_foot_roll(18)
  yaw类:  lw_forearm_yaw(7), rw_forearm_yaw(8), lrw_hip_yaw(11), rw_hip_yaw(12)
  waist:  w_waist_yaw(2)
"""

def _switch_qingyun_z1_A_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor.

    Joint order (19 dof):
     0 lw_shoulder_pitch,  1 rw_shoulder_pitch,  2 w_waist_yaw,
     3 lw_arm_roll,        4 rw_arm_roll,
     5 lrw_hip_pitch,      6 rw_hip_pitch,
     7 lw_forearm_yaw,     8 rw_forearm_yaw,
     9 lrw_hip_roll,      10 rw_hip_roll,
    11 lrw_hip_yaw,       12 rw_hip_yaw,
    13 lrw_knee_pitch,    14 rw_knee_pitch,
    15 lrw_foot_pitch,    16 rw_foot_pitch,
    17 lrw_foot_roll,     18 rw_foot_roll
    """
    joint_data_switched = torch.zeros_like(joint_data)

    # Left indices and corresponding right indices for swapping
    left_indices  = [0, 3, 5, 7,  9, 11, 13, 15, 17]
    right_indices = [1, 4, 6, 8, 10, 12, 14, 16, 18]

    # Waist joint: not swapped, but sign flipped
    joint_data_switched[..., 2] = -joint_data[..., 2]

    # Swap left and right joints
    joint_data_switched[..., left_indices]  = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    # Flip sign for roll joints (after swap)
    roll_indices = [3, 4, 9, 10, 17, 18]
    joint_data_switched[..., roll_indices] *= -1.0

    # Flip sign for yaw joints (after swap)
    yaw_indices = [7, 8, 11, 12]
    joint_data_switched[..., yaw_indices] *= -1.0

    return joint_data_switched


def _switch_qingyun_z1_A_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the key body positions tensor.

    Key body order (6 bodies, 18 dims):
      0 - lrp_foot_roll   (left)
      1 - rp_foot_roll    (right)
      2 - lp_forearm_yaw  (left)
      3 - rp_forearm_yaw  (right)
      4 - lp_arm_roll     (left)
      5 - rp_arm_roll     (right)

    Bodies are in left/right pairs. For each pair:
      - swap left and right positions
      - flip the y-coordinate to reflect left-right symmetry
    """
    key_body_pos_switched = key_body_pos.clone()
    num_key_bodies = key_body_pos.shape[-1] // 3

    for i in range(num_key_bodies // 2):
        left_idx = i * 2
        right_idx = i * 2 + 1

        # Swap left and right key body positions
        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3]   = key_body_pos[..., right_idx * 3 : right_idx * 3 + 3]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[..., left_idx * 3 : left_idx * 3 + 3]

        # Flip the y-coordinate to reflect left-right symmetry
        key_body_pos_switched[..., left_idx * 3 + 1]  *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

    return key_body_pos_switched
