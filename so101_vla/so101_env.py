"""SO-101 pick-and-place environment + expert for VLA training (same idea as ../expert.py for the Panda).

Reuses the scene builder and IK from ../so101_ik.py. The model is built once; every episode only moves
the cube (free joint) and the place marker, instead of rebuilding the scene.

IK is used only to *plan* the 8 joint keyframes (the training labels). The arm itself is driven purely in
joint space by `move_to`, exactly like the VLA runner drives it.

    python so101_env.py                    # watch random expert episodes in the viewer
    python so101_env.py --headless -n 100  # measure expert success rate
"""
import os
import re
import sys
import time
import argparse
from dataclasses import dataclass

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Appended, not prepended: the project root also has generate_dataset.py etc. (Panda versions),
# and this folder's modules must win.
sys.path.append(ROOT)
from so101_ik import (build_model, IK, ARM_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED, CUBE_HALF,  # noqa: E402
                      top_down_rotation, grasp_yaw)

PHASES = ["PRE_GRASP", "GRASP", "CLOSE", "LIFT", "PRE_PLACE", "PLACE", "RELEASE", "RETREAT"]

# TCP heights above the cube centre / place height (same waypoints as so101_ik.py)
HOVER_DZ, GRASP_DZ, LIFT_DZ, PLACE_DZ, RETREAT_DZ = 0.05, 0.002, 0.08, 0.005, 0.06

HOME_POS = np.array([0.18, 0.0, 0.16])
HOME_SEED = np.array([0.0, -1.0, 1.0, 1.2, 0.0])
# Where the arm waits at the start of an episode: the top-down home pose with the base turned 100 deg
# to the left. At HOME_POS itself the arm sits in the middle of the top camera's view and hides the cube.
PARK_PAN = np.radians(100)

# Task randomisation: polar coordinates around the robot base (metres / radians), as in so101_ik --seed.
# Cube and marker sit on opposite sides; which side is which is random.
RADIUS = (0.16, 0.26)
SIDE_ANGLE = (0.15, 0.7)
MIN_CUBE_PLACE_DIST = 0.08

# Top camera moved closer than in so101_ik.py (0.75 m, 50 deg) so the 25 mm cube covers more pixels
TOP_CAM_POS = [0.19, 0.0, 0.50]
TOP_CAM_FOVY = 45.0

PLACE_MARKER_RADIUS = 0.025

# Panda scene look (mujoco_pick_place/franka_emika_panda/mjx_scene.xml + vla_scene.xml): flat dark-blue
# floor, no shadows, no reflections, green cube on a red pad. Simple images for a first model.
PANDA_FLOOR_RGB = (0, 40, 56)              # groundplane texture rgb1="0.0 0.157 0.220", as bytes
PANDA_CUBE_RGBA = (0.0, 1.0, 0.0, 1.0)
PANDA_PAD_RGBA = (1.0, 0.0, 0.0, 1.0)


def apply_panda_look(m):
    """Restyle the so101_ik.py scene to look like the Panda scene. Runtime model edits only (so101_ik.py
    is unchanged); must run before any Renderer / viewer is created, since they upload textures at start."""
    # Floor: the checker texture becomes one flat colour; no reflection
    floor_mat = m.geom_matid[m.geom("floor").id]
    tex = m.mat_texid[floor_mat][m.mat_texid[floor_mat] >= 0][0]
    start, n = m.tex_adr[tex], m.tex_width[tex] * m.tex_height[tex] * m.tex_nchannel[tex]
    m.tex_data[start:start + n] = np.tile(np.array(PANDA_FLOOR_RGB, np.uint8), n // 3)
    m.mat_reflectance[floor_mat] = 0.0

    # Lights: Panda has a headlight, an overhead directional light and an overhead spot light, no shadows
    m.vis.headlight.diffuse[:] = 0.6
    m.vis.headlight.ambient[:] = 0.3
    m.vis.headlight.specular[:] = 0.0
    m.vis.rgba.haze[:] = (0.15, 0.25, 0.35, 1.0)
    directional = np.flatnonzero(m.light_type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)[0]
    spot = np.flatnonzero(m.light_type != mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)[0]
    m.light_pos[spot], m.light_dir[spot] = (0, 0, 2), (0, 0, -1)
    m.light_pos0[spot], m.light_dir0[spot] = (0, 0, 2), (0, 0, -1)
    m.light_diffuse[spot] = 0.7
    for i in (directional, spot):
        m.light_castshadow[i] = 0

    m.geom_rgba[m.geom("cube_geom").id] = PANDA_CUBE_RGBA
    m.geom_rgba[m.geom("place_marker").id] = PANDA_PAD_RGBA


@dataclass
class Keyframe:
    phase: str
    q: np.ndarray       # 5 arm joint angles (rad)
    gripper: float      # gripper command (rad): GRIPPER_OPEN or GRIPPER_CLOSED
    tcp_pos: np.ndarray


def gripper_word(g):
    return "open" if g > (GRIPPER_OPEN + GRIPPER_CLOSED) / 2 else "closed"


def format_action(q, gripper):
    """Assistant answer: 5 joint angles in degrees (0.1 deg) + gripper command."""
    return "j: " + " ".join(f"{v:.1f}" for v in np.degrees(q)) + f" | gripper: {gripper_word(gripper)}"


ACTION_RE = re.compile(r"j:\s*((?:-?\d+(?:\.\d+)?\s+){4}-?\d+(?:\.\d+)?)\s*\|\s*gripper:\s*(open|closed)")


def parse_action(text):
    """'j: a b c d e | gripper: open' -> (q_rad[5], gripper_cmd), or None if malformed."""
    m = ACTION_RE.search(text)
    if m is None:
        return None
    q = np.radians([float(v) for v in m.group(1).split()])
    return q, (GRIPPER_OPEN if m.group(2) == "open" else GRIPPER_CLOSED)


class SO101Env:
    def __init__(self):
        # Placeholder poses; reset() moves the cube and marker every episode
        m = self.model = build_model(np.array([0.2, -0.1, CUBE_HALF]), 0.0, np.array([0.2, 0.1, CUBE_HALF]))
        self.data = mujoco.MjData(m)
        self.ik = IK(m)

        act = {m.actuator(i).name: i for i in range(m.nu)}
        self.arm_act = np.array([act[j] for j in ARM_JOINTS])
        self.grip_act = act["gripper"]
        self.arm_qadr = self.ik.qadr
        self.arm_lo, self.arm_hi = self.ik.lo, self.ik.hi
        self.grip_qadr = m.jnt_qposadr[m.joint("gripper").id]
        self.cube_body = m.body("cube").id
        self.cube_qadr = m.jnt_qposadr[m.joint("cube_joint").id]
        self.marker_geom = m.geom("place_marker").id

        apply_panda_look(m)

        # The IK target sphere is a debugging aid: hide it from the cameras
        target = m.body("ik_target").id
        m.geom_rgba[m.geom_bodyid == target, 3] = 0.0

        top = m.camera("top_camera").id
        m.cam_pos[top] = TOP_CAM_POS
        m.cam_fovy[top] = TOP_CAM_FOVY

        # q_seed: IK seed for planning (keeps labels identical whatever the start pose)
        # q_home: start pose of every episode, parked out of the top camera's view
        self.q_seed, err, _ = self.ik.solve(HOME_POS, top_down_rotation(0.0), HOME_SEED, iters=300)
        assert err < 2e-3, f"home pose unreachable ({err * 1000:.1f} mm)"
        self.q_home = self.q_seed.copy()
        self.q_home[0] = PARK_PAN

    # ── Task setup ─────────────────────────────────────────────────────────
    @staticmethod
    def sample_task(rng):
        sign = rng.choice([-1, 1])  # cube on the right (-y) or left (+y)
        while True:
            r, a = rng.uniform(*RADIUS), sign * -rng.uniform(*SIDE_ANGLE)
            cube_xy = r * np.array([np.cos(a), np.sin(a)])
            r, a = rng.uniform(*RADIUS), sign * rng.uniform(*SIDE_ANGLE)
            place_xy = r * np.array([np.cos(a), np.sin(a)])
            if np.linalg.norm(cube_xy - place_xy) >= MIN_CUBE_PLACE_DIST:
                return {"cube_xy": cube_xy, "cube_yaw": rng.uniform(-np.pi, np.pi), "place_xy": place_xy}

    def reset(self, task):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = self.q_home
        d.ctrl[self.arm_act] = self.q_home
        d.qpos[self.grip_qadr] = GRIPPER_OPEN
        d.ctrl[self.grip_act] = GRIPPER_OPEN

        yaw = task["cube_yaw"]
        d.qpos[self.cube_qadr:self.cube_qadr + 3] = [*task["cube_xy"], CUBE_HALF]
        d.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        m.geom_pos[self.marker_geom] = [*task["place_xy"], 0.0005]

        mujoco.mj_forward(m, d)
        self.step(int(0.3 / m.opt.timestep))  # let the cube settle

    def cube_pose(self):
        w, x, y, z = self.data.xquat[self.cube_body]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return self.data.xpos[self.cube_body].copy(), yaw

    def place_xy(self):
        return self.model.geom_pos[self.marker_geom][:2].copy()

    def arm_q(self):
        return self.data.qpos[self.arm_qadr].copy()

    def forward_tcp(self, q):
        """TCP world position for arm joints q (rad). Kinematics only."""
        return self.ik.fk(q)[0]

    # ── Planning ───────────────────────────────────────────────────────────
    def plan_keyframes(self):
        """8 joint keyframes from the true cube / marker pose. Returns None if a pose is unreachable."""
        cube_pos, cube_yaw = self.cube_pose()
        place_pos = np.array([*self.place_xy(), CUBE_HALF + 0.002])
        gy = grasp_yaw(cube_pos, cube_yaw)
        R_pick = top_down_rotation(gy)
        R_place = top_down_rotation(grasp_yaw(place_pos, gy))
        up = np.array([0.0, 0.0, 1.0])
        targets = [
            ("PRE_GRASP", cube_pos + HOVER_DZ * up,    R_pick,  GRIPPER_OPEN),
            ("GRASP",     cube_pos + GRASP_DZ * up,    R_pick,  GRIPPER_OPEN),
            ("CLOSE",     cube_pos + GRASP_DZ * up,    R_pick,  GRIPPER_CLOSED),
            ("LIFT",      cube_pos + LIFT_DZ * up,     R_pick,  GRIPPER_CLOSED),
            ("PRE_PLACE", place_pos + LIFT_DZ * up,    R_place, GRIPPER_CLOSED),
            ("PLACE",     place_pos + PLACE_DZ * up,   R_place, GRIPPER_CLOSED),
            ("RELEASE",   place_pos + PLACE_DZ * up,   R_place, GRIPPER_OPEN),
            ("RETREAT",   place_pos + RETREAT_DZ * up, R_place, GRIPPER_OPEN),
        ]
        # Seeded from home and chained like so101_ik.py -> deterministic, consistent labels
        keyframes, q_seed = [], self.q_seed.copy()
        for phase, pos, R, grip in targets:
            q_seed, err_pos, _ = self.ik.solve(pos, R, q_seed, iters=300)
            if err_pos > 2e-3:
                return None
            keyframes.append(Keyframe(phase, q_seed.copy(), grip, pos))
        return keyframes

    # ── Joint-space execution (same path the VLA runner uses) ─────────────
    def step(self, n, viewer=None):
        dt = self.model.opt.timestep
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None and i % 10 == 0:
                viewer.sync()
                time.sleep(10 * dt)

    def move_to(self, q_target, gripper, max_speed=1.0, viewer=None):
        """Smooth joint-space interpolation from the current command to q_target, then hold
        (longer when the gripper changes so the jaw can close / open)."""
        d, dt = self.data, self.model.opt.timestep
        q_start = d.ctrl[self.arm_act].copy()
        q_target = np.clip(q_target, self.arm_lo, self.arm_hi)
        gripper_changed = abs(d.ctrl[self.grip_act] - gripper) > 1e-6
        d.ctrl[self.grip_act] = gripper

        duration = max(np.max(np.abs(q_target - q_start)) / max_speed, 0.4)
        n = int(duration / dt)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)  # smoothstep
            d.ctrl[self.arm_act] = q_start + a * (q_target - q_start)
            self.step(1, viewer)
        self.step(int((0.8 if gripper_changed else 0.2) / dt), viewer)

    def is_success(self):
        cube_pos, _ = self.cube_pose()
        on_marker = np.linalg.norm(cube_pos[:2] - self.place_xy()) < PLACE_MARKER_RADIUS
        resting = cube_pos[2] < CUBE_HALF + 0.005
        return bool(on_marker and resting)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    env = SO101Env()
    rng = np.random.default_rng(args.seed)
    successes = 0

    def run(viewer=None):
        nonlocal successes
        for ep in range(args.episodes):
            env.reset(env.sample_task(rng))
            keyframes = env.plan_keyframes()
            if keyframes is None:
                print(f"ep {ep}: IK failed")
                continue
            for kf in keyframes:
                env.move_to(kf.q, kf.gripper, viewer=viewer)
                if viewer is not None and not viewer.is_running():
                    return
            ok = env.is_success()
            successes += ok
            print(f"ep {ep}: {'success' if ok else 'FAIL'}")

    if args.headless:
        run()
    else:
        from mujoco import viewer
        with viewer.launch_passive(env.model, env.data) as v:
            v.cam.lookat[:] = [0.18, 0.0, 0.05]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.7, 150, -25
            run(v)
    print(f"success {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
