"""Reach task for the SO-101 while wrist_roll and the gripper are out of service: move the gripper to just
above the white block. Same scene, calibrated top camera and randomisation as real_env.py.

Phases (one model answer each):
    HOVER : wrist camera 15 cm above the block top, looking down
    REACH : wrist camera 6 cm above the block top: the block fills the wrist view, as in the real reach
            captures (the webcam looks 50 deg off the jaw, so pointing the jaw down would show the robot base)
The model outputs the 4 working joints:  "j: <shoulder_pan> <shoulder_lift> <elbow_flex> <wrist_flex>".
wrist_roll stays at WRIST_ROLL_FIXED and the gripper at GRIPPER_FIXED, in the sim and on the real arm.

    python reach_task.py                    # watch random expert episodes
    python reach_task.py --headless -n 200  # expert success rate
"""
import re
import argparse

import numpy as np
import mujoco

from real_env import (RealSetupEnv, Keyframe, IK, ARM_JOINTS, HOME_POS, HOME_SEED, PARK_PAN, MAT_TOP,
                      BLOCK_HALF, top_down_rotation)
from so101_ik import GRIPPER_OPEN
from real_arm_client import load_joint_map

REACH_PHASES = ["HOVER", "REACH"]
REACH_JOINTS = ARM_JOINTS[:4]          # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex
WRIST_ROLL_FIXED = 0.0                 # rad: middle of its calibrated range (set the real one there)
GRIPPER_FIXED = GRIPPER_OPEN           # the jaw isn't driven; keep the sim jaw where the real one sits
INSTRUCTION = "Move the gripper just above the white block."

BLOCK_TOP = MAT_TOP + 2 * BLOCK_HALF[2]
PHASE_DZ = {"HOVER": 0.15, "REACH": 0.06}    # wrist camera above the block top (jaw tip stays >= 5 cm above)
REACH_RADIUS = (0.18, 0.32)                  # reachable with wrist_roll fixed (all directions)
MAX_TILT_DEG = 35.0                          # 4 joints can't always point straight down; fine for reaching
SUCCESS_TOL = 0.015                          # wrist camera within 1.5 cm of the REACH target
NOTE_PARKED_XY = np.array([-0.40, 0.0])      # the unused green note: on the table behind the robot
# The real arm's limits (deg, real units) as arm_server.py prints them at start-up; the sim ranges are wider
# for some joints, so the IK is kept inside these (mapped through real_joint_map.json, 1 deg extra margin)
REAL_LIMITS_DEG = {"shoulder_pan": (-92.6, 92.6), "shoulder_lift": (-2.0, 195.0), "elbow_flex": (-185.0, 2.0),
                   "wrist_flex": (-97.6, 97.6)}


def format_reach_action(q4):
    return "j: " + " ".join(f"{v:.1f}" for v in np.degrees(q4[:4]))


REACH_RE = re.compile(r"j:\s*((?:-?\d+(?:\.\d+)?\s+){3}-?\d+(?:\.\d+)?)")


def parse_reach_action(text):
    """'j: a b c d' -> q (rad, 4 joints), or None if malformed."""
    m = REACH_RE.search(text)
    return None if m is None else np.radians([float(v) for v in m.group(1).split()])


def build_reach_prompt(phase_idx):
    return (f"Task: {INSTRUCTION}\n"
            f"Phase {phase_idx + 1}/{len(REACH_PHASES)}: {REACH_PHASES[phase_idx]}\n"
            "Output the target joint angles j1..j4 in degrees.")


class ReachEnv(RealSetupEnv):
    def __init__(self, randomize=True):
        super().__init__(randomize, as_built=True)       # the real arm: no moving jaw, zip-tied webcam
        # Targets the wrist camera (site at the lens); its private data keeps wrist_roll at 0
        self.ik4 = IK(self.model, site="wrist_cam", joints=REACH_JOINTS)
        jm = load_joint_map()
        for i, j in enumerate(REACH_JOINTS):
            lo, hi = sorted((np.array(REAL_LIMITS_DEG[j]) - jm[j]["offset_deg"]) * jm[j]["sign"])
            self.ik4.lo[i] = max(self.ik4.lo[i], np.radians(lo + 1.0))
            self.ik4.hi[i] = min(self.ik4.hi[i], np.radians(hi - 1.0))
        q4, err, _ = self.ik4.solve(HOME_POS, top_down_rotation(0.0), HOME_SEED[:4], iters=300)
        assert err < 2e-3, f"home pose unreachable ({err * 1000:.1f} mm)"
        self.q_seed4 = q4
        self.q_seed = self.full_q(q4)                     # safe pose above the mat
        self.q_home = self.full_q(q4)
        self.q_home[0] = PARK_PAN                         # start pose, out of the top camera's view
        self.block_start = None
        # No green note on the real mat for reach: hidden (group 3 isn't rendered), parked behind the robot
        self.model.geom_group[self.model.geom("green_note_geom").id] = 3

    @staticmethod
    def full_q(q4):
        return np.array([*q4[:4], WRIST_ROLL_FIXED])

    def sample_task(self, rng):
        task = super().sample_task(rng)
        task["note_xy"] = NOTE_PARKED_XY
        while True:
            r, a = rng.uniform(*REACH_RADIUS), rng.uniform(-np.pi / 2, np.pi / 2)
            xy = r * np.array([np.cos(a), np.sin(a)])
            if self._in_mat(xy):
                task["block_xy"] = xy
                return task

    def reset(self, task, rng=None):
        super().reset(task, rng)
        d = self.data
        d.ctrl[self.grip_act] = GRIPPER_FIXED
        self.step(int(0.2 / self.model.opt.timestep))
        self.block_start = self.block_pose()[0]

    def cam_point(self):
        return self.data.site_xpos[self.ik4.site].copy()

    def forward_cam(self, q):
        """Wrist camera position for joint angles q (kinematics only)."""
        return self.ik4.fk(q[:4])[0]

    def plan_keyframes(self):
        """HOVER and REACH keyframes: wrist camera above the true block, looking down. None if out of reach."""
        block_pos, _ = self.block_pose()
        R = top_down_rotation(np.arctan2(block_pos[1], block_pos[0]))   # camera down, image up = away from the base
        keyframes, q = [], self.q_seed4.copy()
        for phase in REACH_PHASES:
            pos = np.array([block_pos[0], block_pos[1], BLOCK_TOP + PHASE_DZ[phase]])
            q, err_pos, err_rot = self.ik4.solve(pos, R, q, iters=300)
            if err_pos > 2e-3 or np.degrees(err_rot) > MAX_TILT_DEG:
                return None
            keyframes.append(Keyframe(phase, self.full_q(q), GRIPPER_FIXED, pos))
        return keyframes

    def reach_error(self):
        """Distance from the wrist camera to the REACH target above the block (m)."""
        block_pos, _ = self.block_pose()
        return float(np.linalg.norm(self.cam_point() - [block_pos[0], block_pos[1], BLOCK_TOP + PHASE_DZ["REACH"]]))

    def block_moved(self):
        return float(np.linalg.norm(self.block_pose()[0][:2] - self.block_start[:2]))

    def is_success(self):
        return self.reach_error() < SUCCESS_TOL and self.block_moved() < 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-randomize", action="store_true")
    args = ap.parse_args()

    env = ReachEnv(randomize=not args.no_randomize)
    rng = np.random.default_rng(args.seed)
    stats = {"success": 0, "fail": 0, "out_of_reach": 0}

    def run(viewer=None):
        for ep in range(args.episodes):
            env.reset(env.sample_task(rng), rng)
            kfs = env.plan_keyframes()
            if kfs is None:
                stats["out_of_reach"] += 1
                continue
            for kf in kfs:
                env.move_to(kf.q, kf.gripper, viewer=viewer)
                if viewer is not None and not viewer.is_running():
                    return
            ok = env.is_success()
            stats["success" if ok else "fail"] += 1
            print(f"ep {ep}: {'success' if ok else 'FAIL'}  (wrist camera {env.reach_error() * 1000:.1f} mm from "
                  f"target, block moved {env.block_moved() * 1000:.1f} mm)")

    if args.headless:
        run()
    else:
        from mujoco import viewer
        with viewer.launch_passive(env.model, env.data) as v:
            v.cam.lookat[:] = [0.25, 0.0, 0.05]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9, 200, -30
            run(v)
    print(f"success {stats['success']}/{stats['success'] + stats['fail']} (out of reach: {stats['out_of_reach']})")


if __name__ == "__main__":
    main()
