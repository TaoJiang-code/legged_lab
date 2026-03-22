import gymnasium as gym

from . import agents

from legged_lab.envs import ManagerBasedAmpEnv

gym.register(
    id="LeggedLab-Isaac-AMP-qingyun_z1_A_rev_1_0-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qingyun_z1_A_rev_1_0_amp_env_cfg:qingyun_z1_A_rev_1_0_AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:qingyun_z1_A_rev_1_0_RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-qingyun_z1_A_rev_1_0-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qingyun_z1_A_rev_1_0_amp_env_cfg:qingyun_z1_A_rev_1_0_AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:qingyun_z1_A_rev_1_0_RslRlOnPolicyRunnerAmpCfg",
    },
)
