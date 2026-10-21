"""Expert pick-and-place for the Panda: 6D IK -> 8 joint keyframes -> joint-space execution.

IK is used only to *plan* the keyframes (the labels for VLM training). The robot itself
is driven purely in joint space by `move_to`, exactly like the VLM runner will drive it.

    python expert.py                 # watch random episodes in the viewer
    python expert.py --headless -n 100   # measure expert success rate
"""
import os
import time
import argparse
from dataclasses import dataclass

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(HERE, "mujoco_pick_place/franka_emika_panda/vla_scene.xml")

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

GRIPPER_OPEN, GRIPPER_CLOSED = 0.04, 0.0

CUBE_HALF = 0.025
HOVER_Z = 0.15   # gripper-site height above the cube before descending
GRASP_Z = 0.03   # site height when grasping (finger pads straddle the cube centre)
CARRY_Z = 0.20
PLACE_Z = 0.035  # leaves ~5mm gap under the cube before release

# Task randomisation (metres / radians, world frame; robot base at origin)
CUBE_X, CUBE_Y = (0.40, 0.65), (-0.22, 0.22)
PAD_X, PAD_Y = (0.40, 0.65), (-0.25, 0.25)
YAW_RANGE = (-np.pi / 4, np.pi / 4)
MIN_CUBE_PAD_DIST = 0.15

PHASES = ["PRE_GRASP", "GRASP", "CLOSE", "LIFT", "PRE_PLACE", "PLACE", "RELEASE", "RETREAT"]


@dataclass
class Keyframe:
    phase: str
    q: np.ndarray      # 7 arm joint angles (rad)
    gripper: float     # actuator command: 0.04 open, 0.0 closed
    site_pos: np.ndarray


def wrap_quarter_turn(yaw):
    """A cube looks the same every 90 deg, so map yaw into [-45, 45) deg."""
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def gripper_word(g):
    return "open" if g > GRIPPER_OPEN / 2 else "closed"


def format_action(q, gripper):
    """Assistant answer: 7 joint angles in degrees (0.1 deg) + gripper command."""
    return "j: " + " ".join(f"{v:.1f}" for v in np.degrees(q)) + f" | gripper: {gripper_word(gripper)}"


def parse_action(text):
    """Inverse of format_action. Returns (q_rad[7], gripper_cmd) or raises ValueError."""
    joints_part, grip_part = text.split("|")
    vals = [float(v) for v in joints_part.replace("j:", "").split()]
    if len(vals) != 7:
        raise ValueError(f"expected 7 joint values, got {len(vals)}: {text!r}")
    grip = GRIPPER_OPEN if "open" in grip_part else GRIPPER_CLOSED
    return np.radians(vals), grip


class PickPlaceEnv:
    def __init__(self, scene_path=SCENE_PATH, timestep=0.002):
        # Gravity compensation on the robot (the real Franka controller does this);
        # without it the position servos sag ~2 deg and the gripper lands ~2cm low.
        # Must be set before compiling: MuJoCo skips gravcomp if no body had it at compile time.
        spec = mujoco.MjSpec.from_file(scene_path)
        for body in spec.body("link0").find_all(mujoco.mjtObj.mjOBJ_BODY):
            body.gravcomp = 1.0
        spec.body("link0").gravcomp = 1.0
        m = self.model = spec.compile()
        m.opt.timestep = timestep
        self.data = mujoco.MjData(m)
        self._ik_data = mujoco.MjData(m)  # scratch state for kinematic IK, never stepped

        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_qadr = m.jnt_qposadr[jids]
        self.arm_dadr = m.jnt_dofadr[jids]
        self.arm_range = m.jnt_range[jids].copy()
        self.finger_qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                            for n in ("finger_joint1", "finger_joint2")]
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self.box_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")]
        self.pad_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "place_pad")]

        # "Pointing down" reference orientation = gripper site frame at HOME_Q
        self._set_ik_q(HOME_Q)
        self.down_R = self._ik_data.site_xmat[self.site_id].reshape(3, 3).copy()

    # ── Task setup ─────────────────────────────────────────────────────────
    @staticmethod
    def sample_task(rng):
        cube_xy = np.array([rng.uniform(*CUBE_X), rng.uniform(*CUBE_Y)])
        while True:
            pad_xy = np.array([rng.uniform(*PAD_X), rng.uniform(*PAD_Y)])
            if np.linalg.norm(pad_xy - cube_xy) >= MIN_CUBE_PAD_DIST:
                break
        return {"cube_xy": cube_xy, "cube_yaw": rng.uniform(*YAW_RANGE), "pad_xy": pad_xy}

    def reset(self, task):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = HOME_Q
        d.qpos[self.finger_qadr] = GRIPPER_OPEN
        d.ctrl[:7] = HOME_Q
        d.ctrl[7] = GRIPPER_OPEN

        yaw = task["cube_yaw"]
        d.qpos[self.box_qadr:self.box_qadr + 3] = [*task["cube_xy"], CUBE_HALF]
        d.qpos[self.box_qadr + 3:self.box_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.mocap_pos[self.pad_mocap] = [*task["pad_xy"], 0.0]

        mujoco.mj_forward(m, d)
        self.step(int(0.2 / m.opt.timestep))  # let the cube settle

    def cube_pose(self):
        """Cube position and (quarter-turn wrapped) yaw, from the simulator."""
        w, x, y, z = self.data.xquat[self.box_body]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return self.data.xpos[self.box_body].copy(), wrap_quarter_turn(yaw)

    def pad_xy(self):
        return self.data.mocap_pos[self.pad_mocap][:2].copy()

    def arm_q(self):
        return self.data.qpos[self.arm_qadr].copy()

    # ── IK (kinematics only, on scratch data) ──────────────────────────────
    def _set_ik_q(self, q):
        d = self._ik_data
        d.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def forward_site(self, q):
        """Gripper-site world position for arm joints q (rad). Kinematics only."""
        self._set_ik_q(q)
        return self._ik_data.site_xpos[self.site_id].copy()

    def solve_ik(self, target_pos, target_R, q_seed=HOME_Q, max_iters=300,
                 damping=1e-4, null_gain=0.1, pos_tol=5e-4, rot_tol=5e-3):
        """Damped least-squares 6D IK. Always seeded from the same pose with a
        null-space pull toward HOME_Q, so nearby targets give nearby joint angles
        (consistent labels for the 7-DOF redundancy)."""
        d, s = self._ik_data, self.site_id
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        q = np.array(q_seed, dtype=float)
        for _ in range(max_iters):
            self._set_ik_q(q)
            R = d.site_xmat[s].reshape(3, 3)
            e_pos = target_pos - d.site_xpos[s]
            e_rot = 0.5 * sum(np.cross(R[:, k], target_R[:, k]) for k in range(3))
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                return q, True
            mujoco.mj_jacSite(self.model, d, jacp, jacr, s)
            J = np.vstack([jacp, jacr])[:, self.arm_dadr]                  # 6x7
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))   # 7x6
            dq = J_pinv @ np.concatenate([e_pos, e_rot])
            dq += null_gain * (np.eye(7) - J_pinv @ J) @ (HOME_Q - q)
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.arm_range[:, 0], self.arm_range[:, 1])
        return q, False

    # ── Planning ───────────────────────────────────────────────────────────
    def plan_keyframes(self):
        """8 joint keyframes from the true cube/pad pose. Returns None if IK fails."""
        cube_pos, cube_yaw = self.cube_pose()
        R = rot_z(cube_yaw) @ self.down_R
        cx, cy = cube_pos[:2]
        px, py = self.pad_xy()
        targets = [
            ("PRE_GRASP", (cx, cy, HOVER_Z), GRIPPER_OPEN),
            ("GRASP",     (cx, cy, GRASP_Z), GRIPPER_OPEN),
            ("CLOSE",     (cx, cy, GRASP_Z), GRIPPER_CLOSED),
            ("LIFT",      (cx, cy, CARRY_Z), GRIPPER_CLOSED),
            ("PRE_PLACE", (px, py, CARRY_Z), GRIPPER_CLOSED),
            ("PLACE",     (px, py, PLACE_Z), GRIPPER_CLOSED),
            ("RELEASE",   (px, py, PLACE_Z), GRIPPER_OPEN),
            ("RETREAT",   (px, py, CARRY_Z), GRIPPER_OPEN),
        ]
        keyframes = []
        for phase, pos, grip in targets:
            pos = np.array(pos)
            q, ok = self.solve_ik(pos, R)
            if not ok:
                return None
            keyframes.append(Keyframe(phase, q, grip, pos))
        return keyframes

    # ── Joint-space execution (same path the VLM runner uses) ─────────────
    def step(self, n, viewer=None):
        dt = self.model.opt.timestep
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None and i % 10 == 0:
                viewer.sync()
                time.sleep(10 * dt)

    def move_to(self, q_target, gripper, max_speed=1.0, viewer=None):
        """Smooth joint-space interpolation from the current command to q_target,
        then hold (longer when the gripper changes so the fingers can close/open)."""
        d, dt = self.data, self.model.opt.timestep
        q_start = d.ctrl[:7].copy()
        q_target = np.clip(q_target, self.arm_range[:, 0], self.arm_range[:, 1])
        gripper_changed = abs(d.ctrl[7] - gripper) > 1e-6
        d.ctrl[7] = gripper

        duration = max(np.max(np.abs(q_target - q_start)) / max_speed, 0.3)
        n = int(duration / dt)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)  # smoothstep
            d.ctrl[:7] = q_start + a * (q_target - q_start)
            self.step(1, viewer)
        self.step(int((0.6 if gripper_changed else 0.2) / dt), viewer)

    def is_success(self, tol=0.03):
        cube_pos, _ = self.cube_pose()
        on_pad = np.linalg.norm(cube_pos[:2] - self.pad_xy()) < tol
        resting = cube_pos[2] < CUBE_HALF + 0.01
        return bool(on_pad and resting)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    env = PickPlaceEnv()
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
            run(v)
    print(f"success {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
