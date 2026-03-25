import argparse
import pickle
from pathlib import Path

import yaml

from isaaclab.app import AppLauncher


ROBOT_CHOICES = ["g1", "elf3_lite", "qingyun_z1", "qingyun_z1_A", "qingyun_z1_A_rev_1_0"]

parser = argparse.ArgumentParser(description="Visualize left-right symmetry for supported robots.")
parser.add_argument("--robot", type=str, required=True, choices=ROBOT_CHOICES, help="Robot name.")
parser.add_argument(
    "--mode",
    type=str,
    default="pose",
    choices=["pose", "motion", "table"],
    help="pose: synthetic asymmetric pose, motion: replay a retargeted motion, table: print mirrored joint values.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Path to a retargeted motion .pkl file. If omitted in motion mode, the first file under the default motion dir is used.",
)
parser.add_argument("--frame", type=int, default=0, help="Frame index for table mode.")
parser.add_argument(
    "--fps",
    type=float,
    default=None,
    help="Playback/render fps. Defaults to the motion fps in motion mode, otherwise 30.",
)
parser.add_argument("--pose_scale", type=float, default=0.2, help="Amplitude of the synthetic asymmetric pose.")
parser.add_argument("--spacing", type=float, default=1.5, help="Distance between the original and mirrored robots.")
parser.add_argument("--floor_height", type=float, default=0.0, help="Ground plane height in world Z.")
parser.add_argument(
    "--show_markers",
    action="store_true",
    default=False,
    help="Show key-body markers if the retarget config provides lab_key_body_names.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.mode == "table":
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils  # type: ignore
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

ROBOT_REGISTRY = {
    "g1": {
        "asset_import": ("legged_lab.assets.unitree", "UNITREE_G1_29DOF_CFG"),
        "symmetry_import": ("legged_lab.tasks.locomotion.amp.mdp.symmetry.g1", None),
        "retarget_yaml": "scripts/tools/retarget/config/g1_29dof.yaml",
        "motion_dir": "source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run",
    },
    "elf3_lite": {
        "asset_import": ("legged_lab.assets.elf3", "ELF3LITE_CFG"),
        "symmetry_import": ("legged_lab.tasks.locomotion.amp.mdp.symmetry.elf3_lite", None),
        "retarget_yaml": "scripts/tools/retarget/config/elf3_lite.yaml",
        "motion_dir": "source/legged_lab/legged_lab/data/MotionData/elf3_lite/amp/walk_and_run",
    },
    "qingyun_z1": {
        "asset_import": ("legged_lab.assets.qingyun_z1", "qingyun_z1_CFG"),
        "symmetry_import": ("legged_lab.tasks.locomotion.amp.mdp.symmetry.qingyun_z1", None),
        "retarget_yaml": "scripts/tools/retarget/config/qingyun_z1.yaml",
        "motion_dir": "source/legged_lab/legged_lab/data/MotionData/qingyun_z1/amp/walk_and_run",
    },
    "qingyun_z1_A": {
        "asset_import": ("legged_lab.assets.qingyun_z1_A", "qingyun_z1_A_CFG"),
        "symmetry_import": ("legged_lab.tasks.locomotion.amp.mdp.symmetry.qingyun_z1_A", None),
        "retarget_yaml": "scripts/tools/retarget/config/qingyun_z1_A.yaml",
        "motion_dir": "source/legged_lab/legged_lab/data/MotionData/qingyun_z1_A/amp/walk_and_run",
    },
    "qingyun_z1_A_rev_1_0": {
        "asset_import": ("legged_lab.assets.qingyun_z1_A_rev_1_0", "qingyun_z1_A_rev_1_0_CFG"),
        "symmetry_import": ("legged_lab.tasks.locomotion.amp.mdp.symmetry.qingyun_z1_A_rev_1_0", None),
        "retarget_yaml": "scripts/tools/retarget/config/qingyun_z1_A_rev_1_0.yaml",
        "motion_dir": "source/legged_lab/legged_lab/data/MotionData/qingyun_z1_A_rev_1_0/amp/walk_and_run",
    },
}


def _import_attr(module_name: str, attr_name: str):
    module = __import__(module_name, fromlist=[attr_name] if attr_name else ["*"])
    return getattr(module, attr_name) if attr_name else module


def load_robot_cfg(robot_name: str):
    module_name, attr_name = ROBOT_REGISTRY[robot_name]["asset_import"]
    return _import_attr(module_name, attr_name)


def load_symmetry_module(robot_name: str):
    module_name, _ = ROBOT_REGISTRY[robot_name]["symmetry_import"]
    return _import_attr(module_name, None)


def find_switch_function(symmetry_module, suffix: str):
    for name in dir(symmetry_module):
        if name.startswith("_switch_") and name.endswith(suffix):
            return getattr(symmetry_module, name)
    raise AttributeError(f"Cannot find symmetry helper with suffix '{suffix}' in {symmetry_module.__name__}")


def load_key_body_names(robot_name: str) -> list[str]:
    yaml_path = Path(ROBOT_REGISTRY[robot_name]["retarget_yaml"])
    if not yaml_path.exists():
        return []
    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("lab_key_body_names", [])


def resolve_motion_file(robot_name: str, motion_file: str | None) -> Path:
    if motion_file is not None:
        return Path(motion_file)
    motion_dir = Path(ROBOT_REGISTRY[robot_name]["motion_dir"])
    files = sorted(motion_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl motion file found under {motion_dir}")
    return files[0]


def load_motion_data(robot_name: str, motion_file: str | None):
    motion_path = resolve_motion_file(robot_name, motion_file)
    with motion_path.open("rb") as f:
        motion = pickle.load(f)
    if "dof_pos" not in motion:
        raise KeyError(f"{motion_path} does not contain 'dof_pos'")
    fps = motion.get("fps", 30)
    return motion_path, motion, fps


def create_scene(robot_cfg, spacing: float, floor_height: float):
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg, translation=(0.0, 0.0, floor_height))

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    origin_a = [0.0, -spacing * 0.5, 0.0]
    origin_b = [0.0, spacing * 0.5, 0.0]
    prim_utils.create_prim("/World/OriginA", "Xform", translation=origin_a)
    prim_utils.create_prim("/World/OriginB", "Xform", translation=origin_b)

    robot_a = Articulation(robot_cfg.replace(prim_path="/World/OriginA/Robot"))
    robot_b = Articulation(robot_cfg.replace(prim_path="/World/OriginB/Robot"))
    return robot_a, robot_b, torch.tensor(origin_a), torch.tensor(origin_b)


def build_marker():
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/SymmetryMarkers",
        markers={
            "original": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            ),
            "mirrored": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def generate_asymmetric_pose(default_joint_pos: torch.Tensor, scale: float):
    joint_count = default_joint_pos.shape[1]
    offsets = scale * torch.sin(torch.arange(joint_count, device=default_joint_pos.device, dtype=torch.float32) * 0.7)
    return default_joint_pos + offsets.unsqueeze(0)


def write_pose(robot: Articulation, root_state: torch.Tensor, joint_pos: torch.Tensor):
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)


def update_markers(marker, robot_a, robot_b, key_body_names: list[str]):
    if marker is None or not key_body_names:
        return
    body_names = robot_a.data.body_names
    indices = []
    for name in key_body_names:
        if name in body_names:
            indices.append(body_names.index(name))
    if not indices:
        return
    pos_a = robot_a.data.body_pos_w[0, indices, :]
    pos_b = robot_b.data.body_pos_w[0, indices, :]
    marker_indices = torch.cat(
        [
            torch.zeros(len(indices), dtype=torch.int32, device=pos_a.device),
            torch.ones(len(indices), dtype=torch.int32, device=pos_a.device),
        ]
    )
    marker.visualize(translations=torch.cat([pos_a, pos_b], dim=0), marker_indices=marker_indices)


def print_table(joint_names: list[str], original: torch.Tensor, mirrored: torch.Tensor):
    print("Index | Joint Name | Original | Mirrored")
    print("-" * 64)
    for idx, name in enumerate(joint_names):
        print(f"{idx:>5} | {name:<24} | {original[idx]:>8.4f} | {mirrored[idx]:>8.4f}")


def main():
    robot_cfg = load_robot_cfg(args_cli.robot)
    symmetry_module = load_symmetry_module(args_cli.robot)
    switch_joints = find_switch_function(symmetry_module, "joints_left_right")
    key_body_names = load_key_body_names(args_cli.robot)

    fps = 30.0
    if args_cli.mode == "motion":
        motion_path, motion, motion_fps = load_motion_data(args_cli.robot, args_cli.motion_file)
        fps = args_cli.fps if args_cli.fps is not None else motion_fps
        print(f"[INFO] Using motion file: {motion_path}")
        dof_pos_seq = torch.tensor(motion["dof_pos"], dtype=torch.float32)
        root_pos_seq = torch.tensor(motion.get("root_pos", []), dtype=torch.float32) if "root_pos" in motion else None
        root_rot_seq = torch.tensor(motion.get("root_rot", []), dtype=torch.float32) if "root_rot" in motion else None
    else:
        fps = args_cli.fps if args_cli.fps is not None else fps
        dof_pos_seq = None
        root_pos_seq = None
        root_rot_seq = None

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / fps, device=args_cli.device))
    sim.set_camera_view(eye=[2.8, 2.8, 1.8], target=[0.0, 0.0, 0.9])
    robot_a, robot_b, origin_a, origin_b = create_scene(robot_cfg, args_cli.spacing, args_cli.floor_height)
    marker = build_marker() if args_cli.show_markers else None

    sim.reset()
    robot_a.update(sim.get_physics_dt())
    robot_b.update(sim.get_physics_dt())

    origin_a = origin_a.to(sim.device)
    origin_b = origin_b.to(sim.device)

    default_root_a = robot_a.data.default_root_state.clone()
    default_root_b = robot_b.data.default_root_state.clone()
    default_root_a[:, :3] = origin_a + default_root_a[:, :3]
    default_root_b[:, :3] = origin_b + default_root_b[:, :3]
    default_joint_pos = robot_a.data.default_joint_pos.clone()
    joint_names = list(robot_a.data.joint_names)

    if args_cli.mode == "table":
        if dof_pos_seq is not None:
            frame_idx = max(0, min(args_cli.frame, dof_pos_seq.shape[0] - 1))
            original_joint_pos = dof_pos_seq[frame_idx : frame_idx + 1].to(sim.device)
        else:
            original_joint_pos = generate_asymmetric_pose(default_joint_pos, args_cli.pose_scale)
        mirrored_joint_pos = switch_joints(original_joint_pos).cpu()[0]
        print_table(joint_names, original_joint_pos.cpu()[0], mirrored_joint_pos)
        return

    print(f"[INFO] Joint count: {len(joint_names)}")
    print(f"[INFO] Key-body markers: {'on' if marker is not None and key_body_names else 'off'}")

    if args_cli.mode == "pose":
        original_joint_pos = generate_asymmetric_pose(default_joint_pos, args_cli.pose_scale)
        mirrored_joint_pos = switch_joints(original_joint_pos.clone())
        while simulation_app.is_running():
            write_pose(robot_a, default_root_a, original_joint_pos)
            write_pose(robot_b, default_root_b, mirrored_joint_pos)
            sim.render()
            robot_a.update(sim.get_physics_dt())
            robot_b.update(sim.get_physics_dt())
            update_markers(marker, robot_a, robot_b, key_body_names)
        return

    frame_count = dof_pos_seq.shape[0]
    print(f"[INFO] Motion frames: {frame_count}, fps: {fps}")
    frame = 0
    while simulation_app.is_running():
        frame_idx = frame % frame_count
        original_joint_pos = dof_pos_seq[frame_idx : frame_idx + 1].to(sim.device)
        mirrored_joint_pos = switch_joints(original_joint_pos.clone())

        root_state_a = default_root_a.clone()
        root_state_b = default_root_b.clone()
        if root_pos_seq is not None and root_pos_seq.numel() > 0:
            root_state_a[:, :3] = origin_a + root_pos_seq[frame_idx : frame_idx + 1].to(sim.device)
            root_state_b[:, :3] = origin_b + root_pos_seq[frame_idx : frame_idx + 1].to(sim.device)
        if root_rot_seq is not None and root_rot_seq.numel() > 0:
            root_state_a[:, 3:7] = root_rot_seq[frame_idx : frame_idx + 1].to(sim.device)
            root_state_b[:, 3:7] = root_rot_seq[frame_idx : frame_idx + 1].to(sim.device)

        write_pose(robot_a, root_state_a, original_joint_pos)
        write_pose(robot_b, root_state_b, mirrored_joint_pos)
        sim.render()
        robot_a.update(sim.get_physics_dt())
        robot_b.update(sim.get_physics_dt())
        update_markers(marker, robot_a, robot_b, key_body_names)
        frame += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

# --robot
# 可选：g1、elf3_lite、qingyun_z1、qingyun_z1_A、qingyun_z1_A_rev_1_0

# --mode
# 可选：pose、motion、table
# pose：显示一组自动生成的左右不对称姿态
# motion：播放 retarget 后动作并显示镜像
# table：打印原始关节值和镜像关节值

# --motion_file
# 只在 motion 模式最有用
# 指定某个 .pkl 动作文件；不填就自动去默认 motion 目录里取第一个

# --frame
# 主要给 table 模式用
# 指定打印 motion 的第几帧

# --pose_scale
# 给 pose 模式用
# 控制自动生成的非对称关节偏移幅度，默认 0.2

# --spacing
# 控制左右两个机器人之间的距离，默认 1.5

# --show_markers
# 打开 key body marker 可视化

# 另外还继承了 AppLauncher 的常用参数，通常至少可以用：

# --headless
# --device
# 以及 Isaac Sim/Isaac Lab 自带的一些启动参数
 
# --floor_height
# 控制地面高度，默认为 0.0，即地面在世界坐标系的 Z=0 平面上

# Example:
# python scripts/tools/visualize_symmetry.py --robot qingyun_z1_A_rev_1_0 --mode pose
# python scripts/tools/visualize_symmetry.py --robot g1 --mode motion --motion_file source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/B10_-__Walk_turn_left_45_stageii.pkl
# python scripts/tools/visualize_symmetry.py --robot elf3_lite --mode table




# python scripts/tools/visualize_symmetry.py \
# --robot qingyun_z1_A_rev_1_0 \
# --mode motion \
# --spacing 1.5 \
# --show_markers \
# --floor_height -0.3 \
# --motion_file temp/lab_data/run1_run1_subject2.pkl
