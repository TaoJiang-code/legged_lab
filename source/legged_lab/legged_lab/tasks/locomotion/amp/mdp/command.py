from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg , UniformVelocityCommand
# from isaaclab.managers import CommandTermCfg
# from dataclasses import MISSING

# class UniformVelocityCommandCfg(CommandTermCfg):
#     """Configuration for the uniform velocity command generator."""

#     class_type: type = UniformVelocityCommand

#     asset_name: str = MISSING
#     """Name of the asset in the environment for which the commands are generated."""

#     heading_command: bool = False
#     """Whether to use heading command or angular velocity command. Defaults to False.

#     If True, the angular velocity command is computed from the heading error, where the
#     target heading is sampled uniformly from provided range. Otherwise, the angular velocity
#     command is sampled uniformly from provided range.
#     """

#     heading_control_stiffness: float = 1.0
#     """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

#     rel_standing_envs: float = 0.0
#     """The sampled probability of environments that should be standing still. Defaults to 0.0."""

#     rel_heading_envs: float = 1.0
#     """The sampled probability of environments where the robots follow the heading-based angular velocity command
#     (the others follow the sampled angular velocity command). Defaults to 1.0.

#     This parameter is only used if :attr:`heading_command` is True.
#     """
