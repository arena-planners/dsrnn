"""CrowdNav_DSRNN (DS-RNN, unicycle) wrapper for the arena_planners bridge.

Reconstructs, outside the upstream ``crowd_sim`` simulator, the SRNN observation
dict (robot_node + temporal_edges + spatial_edges) exactly as
``crowd_sim/envs/crowd_sim_dict.py::generate_ob`` builds it, runs the recurrent
policy, and maps the unicycle action to a differential-drive ``[v, omega]`` twist.

No observation normalization is applied: upstream ``test.py`` evaluates with
``ob_rms = False`` and ships no running mean/var stats, so the trained policy
consumes raw observations.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from arena_planners.sdk import load_manifest, main_loop

from dsrnn_config import Config
from srnn_net import Policy

_HERE = pathlib.Path(__file__).parent
_MODEL_DIR = _HERE / "model"
_CHECKPOINT = _MODEL_DIR / "55554.pt"

# Unicycle action limits / dynamics from the upstream unicycle config + clip_action.
_DELTA_V_CLIP: float = 0.1          # crowd_nav/policy/srnn.py: |raw[0]| <= 0.1 (change of v)
_DELTA_THETA_CLIP: float = 0.1      # crowd_nav/policy/srnn.py: |raw[1]| <= 0.1 (change of theta)
_TIME_STEP: float = 0.25            # config.env.time_step
_RADIUS: float = 0.3                # config.robot.radius
_V_PREF: float = 1.0                # config.robot.v_pref


class _Runner:
    def __init__(self) -> None:
        self.config = Config()
        self.human_num = int(self.config.sim.human_num)

        self.policy = Policy(obs_shape=None, action_dim=2, base_kwargs=self.config)
        self.policy.base.nenv = 1  # acting model: single env
        state = torch.load(str(_CHECKPOINT), map_location="cpu")
        self.policy.load_state_dict(state, strict=True)
        self.policy.eval()

        node_size = int(self.config.SRNN.human_node_rnn_size)
        edge_size = int(self.config.SRNN.human_human_edge_rnn_size)
        self._node_size = node_size
        self._edge_size = edge_size
        self.reset_state()

    def reset_state(self) -> None:
        # eval hidden states: human_node_rnn (1, node_size), human_human_edge_rnn (human_num+1, edge_size).
        self.rnn_hxs = {
            "human_node_rnn": torch.zeros(1, 1, self._node_size),
            "human_human_edge_rnn": torch.zeros(1, self.human_num + 1, self._edge_size),
        }
        # masks=0 on the very first step (episode start), 1.0 thereafter — matches eval loop.
        self.masks = torch.zeros(1, 1)
        # accumulated absolute velocity (crowd_sim_dict.step: desiredVelocity[0]).
        self.desired_v = 0.0

    def act(self, robot_node, temporal_edges, spatial_edges):
        # Leading nenv=1 dim matches the VecEnv obs batching that reshapeT() expects:
        # robot_node (nenv,1,7), temporal_edges (nenv,1,2), spatial_edges (nenv,human_num,2).
        inputs = {
            "robot_node": torch.tensor(robot_node, dtype=torch.float32).reshape(1, 1, 7),
            "temporal_edges": torch.tensor(temporal_edges, dtype=torch.float32).reshape(1, 1, 2),
            "spatial_edges": torch.tensor(spatial_edges, dtype=torch.float32).reshape(
                1, self.human_num, 2
            ),
        }
        with torch.no_grad():
            _, action, self.rnn_hxs = self.policy.act(
                inputs, self.rnn_hxs, self.masks, deterministic=True
            )
        self.masks = torch.ones(1, 1)
        return action.reshape(-1).numpy()


_runner: _Runner | None = None


def _get_runner() -> _Runner:
    global _runner
    if _runner is None:
        _runner = _Runner()
    return _runner


def step(features: dict) -> list[float]:
    """Map the bridge feature dict to a unicycle [v, omega] twist."""
    runner = _get_runner()

    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    goal_pose = features.get("goal_pose")
    if robot_pose is None or goal_pose is None:
        return [0.0, 0.0]

    px, py = float(robot_pose[0]), float(robot_pose[1])
    # crowd_sim wraps theta to [0, 2pi) (agent.step), tf yaw is (-pi, pi].
    theta = float(robot_pose[2]) % (2.0 * np.pi)
    if robot_state is not None and len(robot_state) >= 4:
        vx, vy = float(robot_state[2]), float(robot_state[3])
    else:
        vx, vy = 0.0, 0.0
    gx, gy = float(goal_pose[0]), float(goal_pose[1])

    # robot_node: px, py, radius, gx, gy, v_pref, theta  (get_full_state_list_noV), robot-centred
    robot_node = [0.0, 0.0, _RADIUS, gx - px, gy - py, _V_PREF, theta]
    # temporal_edges: world-frame robot velocity (upstream robot.vx/vy =
    # v*cos/sin(theta)). The odom twist is body-frame, so rotate body->world.
    temporal_edges = [vx * np.cos(theta) - vy * np.sin(theta),
                      vx * np.sin(theta) + vy * np.cos(theta)]

    # spatial_edges: vector robot->human for each human, fixed at human_num rows.
    peds = features.get("pedestrians")
    rel = []
    if peds is not None:
        for ped in peds:
            hx, hy = float(ped[1]), float(ped[2])
            d2 = (hx - px) ** 2 + (hy - py) ** 2
            rel.append((d2, hx - px, hy - py))
    rel.sort(key=lambda r: r[0])  # nearest first if more than human_num present
    spatial_edges = np.zeros((runner.human_num, 2), dtype=np.float32)
    # absent humans: upstream initializes last_human_states far away (15,15); keep that
    # offset robot-relative so they fall outside FOV/attention influence.
    spatial_edges[:] = (15.0, 15.0)
    for i in range(min(runner.human_num, len(rel))):
        spatial_edges[i] = (rel[i][1], rel[i][2])

    raw = runner.act(robot_node, temporal_edges, spatial_edges)

    # clip_action (unicycle, sim2real branch): change-of-v and change-of-theta.
    delta_v = float(np.clip(raw[0], -_DELTA_V_CLIP, _DELTA_V_CLIP))
    delta_theta = float(np.clip(raw[1], -_DELTA_THETA_CLIP, _DELTA_THETA_CLIP))

    # crowd_sim_dict.step: desiredVelocity accumulates delta_v, clipped to +/- v_pref.
    runner.desired_v = float(np.clip(runner.desired_v + delta_v, -_V_PREF, _V_PREF))
    v = runner.desired_v
    # ActionRot.r is a change-of-heading over one time_step -> angular velocity.
    omega = delta_theta / _TIME_STEP
    return [v, omega]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    runner = _get_runner()
    runner.reset_state()


if __name__ == "__main__":
    manifest = load_manifest(_HERE / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
