import sys
import os 
import argparse
import inspect
import shutil


PRETRAINED_DOMAIN_RAND = {
    "robot_friction_range": [0.1, 1.0],
    "robot_restitution_range": [0.2, 0.8],
    "robot_payload_mass_range": [0.0, 3.0],
    "robot_com_displacement_range": [-0.05, 0.05],
    "robot_motor_strength_range": [0.95, 1.05],
    "robot_motor_offset_range": [-0.005, 0.05],
    "ball_radius_range": [0.45, 0.55],
    "ball_mass_range": [1.0, 3.0],
    "ball_friction_range": [0.5, 2.5],
    "ball_restitution_range": [0.4, 0.9],
    "ball_compliance_range": [0.0, 1.0],
    "ball_drag_range": [0.1, 0.5],
    "terrain_ground_friction_range": [0.2, 0.8],
    "terrain_ground_restitution_range": [0.0, 0.5],
    "terrain_tile_roughness_range": [0.005, 0.02],
    "robot_push_vel_range": [0.1, 0.4],
    "ball_push_vel_range": [0.1, 0.4],
    "gravity_range": [-0.1, 0.1],
}


def apply_domain_rand_profile(Cfg, profile):
    if profile == "repo":
        return
    if profile != "pretrained":
        raise ValueError(f"Invalid domain_rand_profile: {profile}")
    for key, value in PRETRAINED_DOMAIN_RAND.items():
        setattr(Cfg.domain_rand, key, value)


if "dirs_exist_ok" not in inspect.signature(shutil.copytree).parameters:
    _copytree = shutil.copytree

    def _copytree_compat(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
                         ignore_dangling_symlinks=False, dirs_exist_ok=False):
        if not dirs_exist_ok or not os.path.isdir(dst):
            return _copytree(src, dst, symlinks=symlinks, ignore=ignore,
                             copy_function=copy_function,
                             ignore_dangling_symlinks=ignore_dangling_symlinks)
        names = os.listdir(src)
        ignored_names = set(ignore(src, names)) if ignore is not None else set()
        for name in names:
            if name in ignored_names:
                continue
            src_name = os.path.join(src, name)
            dst_name = os.path.join(dst, name)
            if os.path.isdir(src_name):
                _copytree_compat(src_name, dst_name, symlinks=symlinks, ignore=ignore,
                                 copy_function=copy_function,
                                 ignore_dangling_symlinks=ignore_dangling_symlinks,
                                 dirs_exist_ok=True)
            else:
                copy_function(src_name, dst_name)
        return dst

    shutil.copytree = _copytree_compat

def train_go1(
    iterations,
    dr_config,
    robot="go1",
    headless=True,
    resume_path=None,
    no_wandb=False,
    wandb_group=None,
    num_envs=None,
    record_video=True,
    save_video_interval=None,
    save_interval=None,
    num_steps_per_env=None,
    domain_rand_profile="repo",
    physx_profile=None,
    resume_checkpoint="ac_weights_last.pt",
    early_stop=True,
    early_stop_warmup_iterations=1000,
    early_stop_patience_iterations=800,
    early_stop_min_delta=0.01,
    early_stop_ema_alpha=0.1,
):

    import isaacgym
    assert isaacgym
    import torch

    from globe_walking.go1_gym.envs.base.legged_robot_config import Cfg
    from globe_walking.go1_gym.envs.go1.go1_config import config_go1
    from globe_walking.go1_gym.envs.go2.go2_config import config_go2
    from globe_walking.go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from globe_walking.go1_gym_learn.ppo_cse import Runner
    from globe_walking.go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from globe_walking.go1_gym_learn.ppo_cse.actor_critic import AC_Args
    from globe_walking.go1_gym_learn.ppo_cse.ppo import PPO_Args
    from globe_walking.go1_gym_learn.ppo_cse import RunnerArgs

    from ml_logger import logger

    if dr_config == "eureka":
        Cfg.env = Cfg.env_full
        Cfg.sensors = Cfg.sensors_full
        Cfg.terrain = Cfg.terrain_full
        Cfg.domain_rand = Cfg.domain_rand_eureka
        Cfg.sim.physx = Cfg.sim.physx_full
    elif dr_config == "off":
        Cfg.env = Cfg.env_mini
        Cfg.sensors = Cfg.sensors_mini
        Cfg.terrain = Cfg.terrain_mini
        Cfg.domain_rand = Cfg.domain_rand_off
        Cfg.sim.physx = Cfg.sim.physx_mini
    else:
        raise ValueError(f"Invalid dr_config: {dr_config}")

    if physx_profile is not None:
        if physx_profile == "mini":
            Cfg.sim.physx = Cfg.sim.physx_mini
        elif physx_profile == "full":
            Cfg.sim.physx = Cfg.sim.physx_full
        else:
            raise ValueError(f"Invalid physx_profile: {physx_profile}")

    robot_configs = {
        "go1": config_go1,
        "go2": config_go2,
    }
    if robot not in robot_configs:
        raise ValueError(f"Invalid robot: {robot}")
    robot_configs[robot](Cfg)
    apply_domain_rand_profile(Cfg, domain_rand_profile)
    if num_envs is not None:
        Cfg.env.num_envs = int(num_envs)
    Cfg.env.record_video = bool(record_video)

    if resume_path:
        RunnerArgs.resume = True
        RunnerArgs.load_run = resume_path
        RunnerArgs.resume_checkpoint = os.path.join(RunnerArgs.load_run, "checkpoints", resume_checkpoint)

    Cfg.robot.name = robot

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.commands.distributional_commands = True

    if robot == "go1":
        Cfg.control.control_type = "actuator_net"

    Cfg.env.num_observation_history = 15

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.balance_gait_distribution = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False

    ###############################
    # globe walking configuration
    ###############################

    # ball parameters
    Cfg.env.add_balls = True

    # sensory observation
    Cfg.commands.num_commands = 0
    Cfg.env.episode_length_s = 40.
    Cfg.env.num_observations = 56

    # terrain configuration
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "boxes_tm"
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.num_border_boxes = 5.0
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False
    Cfg.terrain.center_span = 3
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.difficulty_scale = 1.0
    Cfg.terrain.max_step_height = 0.26
    Cfg.terrain.min_step_run = 0.25
    Cfg.terrain.max_step_run = 0.4
    Cfg.terrain.max_init_terrain_level = 1
    Cfg.terrain.x_init_range = 0.05
    Cfg.terrain.y_init_range = 0.05

    # terminal conditions
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.rewards.reward_container_name = "EurekaReward"
    Cfg.asset.terminate_after_contacts_on = []

    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    RunnerArgs.save_video_interval = 500
    if save_video_interval is not None:
        RunnerArgs.save_video_interval = int(save_video_interval)
    if save_interval is not None:
        RunnerArgs.save_interval = int(save_interval)
    if num_steps_per_env is not None:
        RunnerArgs.num_steps_per_env = int(num_steps_per_env)
    RunnerArgs.early_stop_enabled = bool(early_stop)
    RunnerArgs.early_stop_warmup_iterations = int(early_stop_warmup_iterations)
    RunnerArgs.early_stop_patience_iterations = int(early_stop_patience_iterations)
    RunnerArgs.early_stop_min_delta = float(early_stop_min_delta)
    RunnerArgs.early_stop_ema_alpha = float(early_stop_ema_alpha)
    if not 0.0 < RunnerArgs.early_stop_ema_alpha <= 1.0:
        raise ValueError("--early-stop-ema-alpha must be in (0, 1]")

    import wandb
    if (Cfg.multi_gpu and int(os.getenv("LOCAL_RANK", "0")) == 0) or not Cfg.multi_gpu:
        time_now = logger.utcnow(f'globe_walking/%Y-%m-%d/{Path(__file__).stem}/%H%M%S.%f')
        logger.configure(time_now, root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
        logger.log_text("""
                    charts: 
                    - yKey: train/episode/rew_total/mean
                    xKey: iterations
                    - yKey: train/episode/rew_tracking_lin_vel/mean
                    xKey: iterations
                    - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                    xKey: iterations
                    - yKey: train/episode/rew_action_smoothness_1/mean
                    xKey: iterations
                    - yKey: train/episode/rew_action_smoothness_2/mean
                    xKey: iterations
                    - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                    xKey: iterations
                    - yKey: train/episode/rew_orientation_control/mean
                    xKey: iterations
                    - yKey: train/episode/rew_dof_pos/mean
                    xKey: iterations
                    - yKey: train/episode/command_area_trot/mean
                    xKey: iterations
                    - yKey: train/episode/max_terrain_height/mean
                    xKey: iterations
                    - type: video
                    glob: "videos/*.mp4"
                    - yKey: adaptation_loss/mean
                    xKey: iterations
                    """, filename=".charts.yml", dedent=True)
        logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), RunnerArgs=vars(RunnerArgs),
                        Cfg=vars(Cfg))

        run_name = logger.prefix.split("/")[-1]
        name_prefix = wandb_group + "/" if wandb_group is not None else ""
        wandb.init(
            project="globe_walking",
            entity="upenn-pal",
            name=f"{name_prefix}{run_name}",
            group=wandb_group,
            config={
                "AC_Args": vars(AC_Args),
                "PPO_Args": vars(PPO_Args),
                "RunnerArgs": vars(RunnerArgs),
                "Cfg": vars(Cfg),
            },
            mode=("disabled" if no_wandb else "online")
        )


    if Cfg.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f'cuda:{rank}'
    else:
        device = 'cuda:0'
    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg)

    env = HistoryWrapper(env)
    runner = Runner(env, device=device, multi_gpu=Cfg.multi_gpu)
    runner.learn(num_learning_iterations=int(iterations), init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from globe_walking.go1_gym import MINI_GYM_ROOT_DIR
    from ml_logger import logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-group", type=str)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--save-video-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--num-steps-per-env", type=int, default=None)
    parser.add_argument("--domain-rand-profile", type=str, default="repo", choices=["repo", "pretrained"])
    parser.add_argument("--physx-profile", type=str, default=None, choices=["mini", "full"])
    parser.add_argument("--resume-run", type=str, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default="ac_weights_last.pt")
    parser.add_argument("--robot", type=str, default="go1", choices=["go1", "go2"])
    parser.set_defaults(early_stop=True)
    parser.add_argument("--early-stop", dest="early_stop", action="store_true")
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    parser.add_argument("--early-stop-warmup-iterations", type=int, default=1000)
    parser.add_argument("--early-stop-patience-iterations", type=int, default=800)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.01)
    parser.add_argument("--early-stop-ema-alpha", type=float, default=0.1)

    parser.add_argument("--dr-config", type=str, required=True, choices=["eureka", "off"])
    parser.add_argument("--reward-config", type=str, required=True, choices=["eureka"])
    args = parser.parse_args()

    assert args.reward_config == "eureka", "Only Eureka reward is available"

    resume_path = args.resume_run
    train_go1(
        iterations=args.iterations,
        dr_config=args.dr_config,
        robot=args.robot,
        headless=True,
        resume_path=resume_path,
        no_wandb=args.no_wandb,
        wandb_group=args.wandb_group,
        num_envs=args.num_envs,
        record_video=not args.no_video,
        save_video_interval=args.save_video_interval,
        save_interval=args.save_interval,
        num_steps_per_env=args.num_steps_per_env,
        domain_rand_profile=args.domain_rand_profile,
        physx_profile=args.physx_profile,
        resume_checkpoint=args.resume_checkpoint,
        early_stop=args.early_stop,
        early_stop_warmup_iterations=args.early_stop_warmup_iterations,
        early_stop_patience_iterations=args.early_stop_patience_iterations,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_ema_alpha=args.early_stop_ema_alpha,
    )
