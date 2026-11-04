"""Pick task for the SO-101 as built (04-11-2026: gripper back, wrist_roll taped, webcam on top of the wrist):
pick up the white block, put it back down where it was, let go.

Phases (one model answer each):
    PRE_GRASP : gripper open, straight above the block, jaws across it
    REFINE    : the arm is near PRE_GRASP but a bit off; the prompt also gives the current joint angles and
                the answer is the corrected PRE_GRASP pose. Repeated --refine times (closed loop on the real arm).
    GRASP     : down around the block, open
    CLOSE     : same pose, gripper closed
    LIFT      : 6 cm up
    PLACE     : back down at the same spot
    RELEASE   : gripper open
    RETREAT   : up, open
The model outputs the 4 driven joints and the gripper:  "j: <pan> <lift> <elbow> <wrist_flex> | gripper: open".
wrist_roll isn't driven: the jaws always close across the arm's direction, so the block is placed square to the
arm (its faces facing the robot); the sim samples the block within +-BLOCK_YAW_DEG of that.

    python pick_task.py                    # watch random expert episodes
    python pick_task.py --headless -n 200  # expert success rate
"""
import re
import argparse

import numpy as np

from real_env import (Keyframe, IK, MAT_TOP, BLOCK_HALF, GRASP_Z, HOVER_Z, LIFT_DZ, GRIPPER_CLOSED,
                      AS_BUILT_GRIPPER_OPEN, top_down_rotation)
from reach_task import ReachEnv, REACH_JOINTS, REFINE_DEG

PICK_PHASES = ["PRE_GRASP", "REFINE", "GRASP", "CLOSE", "LIFT", "PLACE", "RELEASE", "RETREAT"]
MOVE_PHASES = [p for p in PICK_PHASES if p != "REFINE"]   # REFINE corrects to the PRE_GRASP target
INSTRUCTION = "Pick up the white block, put it back down and let go."
BLOCK_YAW_DEG = 10.0         # the real block is put square to the arm, roughly
MAX_GRASP_TILT_DEG = 15.0    # jaw axis from straight down, for GRASP / CLOSE / PLACE / RELEASE
PLACE_DZ = 0.003             # block bottom this far above the mat when put back down
# Aim the jaws this far to the arm's left (the fixed finger's side) of the block: the moving finger then pushes
# the block onto the fixed one. Sim: a grasp aimed 5..12+ mm left still lifts it, one 8 mm right doesn't (the
# fixed finger lands on the block), so this centres the window a model's error has to stay in.
GRASP_AIM_LEFT = 0.005
LIFTED_DZ = 0.03             # after LIFT the block must be at least this much higher than at the start
PUT_BACK_TOL = 0.02          # ... and at the end back within this of where it was (xy)
GRIPPER = {"open": AS_BUILT_GRIPPER_OPEN, "closed": GRIPPER_CLOSED}


def gripper_word(g):
    return "open" if g > (GRIPPER["open"] + GRIPPER["closed"]) / 2 else "closed"


def format_pick_action(q4, gripper):
    return "j: " + " ".join(f"{v:.1f}" for v in np.degrees(q4[:4])) + f" | gripper: {gripper_word(gripper)}"


PICK_RE = re.compile(r"j:\s*((?:-?\d+(?:\.\d+)?\s+){3}-?\d+(?:\.\d+)?)\s*\|\s*gripper:\s*(open|closed)")


def parse_pick_action(text):
    """'j: a b c d | gripper: open' -> (q rad, 4 joints; sim gripper command), or None if malformed."""
    m = PICK_RE.search(text)
    if m is None:
        return None
    return np.radians([float(v) for v in m.group(1).split()]), GRIPPER[m.group(2)]


def build_pick_prompt(phase_idx, current_q4=None):
    """current_q4 (rad, 4 joints): where the arm is now; required for REFINE, not given for the other phases."""
    current = ""
    if PICK_PHASES[phase_idx] == "REFINE":
        current = "Current joints: " + " ".join(f"{v:.1f}" for v in np.degrees(current_q4[:4])) + "\n"
    return (f"Task: {INSTRUCTION}\n"
            f"Phase {phase_idx + 1}/{len(PICK_PHASES)}: {PICK_PHASES[phase_idx]}\n"
            f"{current}"
            "Output the target joint angles j1..j4 in degrees and the gripper state.")


class PickEnv(ReachEnv):
    """ReachEnv's arm (4 driven joints, taped wrist_roll, real joint limits), IK on the grasp point (tcp)."""

    def __init__(self, randomize=True):
        super().__init__(randomize)
        self.ik_tcp = IK(self.model, site="tcp", joints=REACH_JOINTS)
        self.ik_tcp.lo[:], self.ik_tcp.hi[:] = self.ik4.lo, self.ik4.hi
        self.ik_tcp.d.qpos[self.roll_qadr] = self.roll
        self.block_lifted = None

    def sample_task(self, rng):
        task = super().sample_task(rng)
        # square to the arm: faces across the radial direction (+-BLOCK_YAW_DEG); a cube looks the same every 90 deg
        az = np.arctan2(task["block_xy"][1], task["block_xy"][0])
        task["block_yaw"] = az + np.radians(rng.uniform(-BLOCK_YAW_DEG, BLOCK_YAW_DEG)) + rng.integers(4) * np.pi / 2
        return task

    def reset(self, task, rng=None):
        super().reset(task, rng)
        self.ik_tcp.d.qpos[self.roll_qadr] = self.roll
        d = self.data
        d.qpos[self.grip_qadr] = d.ctrl[self.grip_act] = GRIPPER["open"]
        self.step(int(0.1 / self.model.opt.timestep))
        self.block_lifted = None

    def _solve(self, pos, q_seed):
        """TCP at pos, jaw axis as close to straight down as the 4 joints allow. The jaws' closing direction
        can't be chosen (wrist_roll is taped): take the one the arm gives there, so only the tilt is an error."""
        q, _, _ = self.ik_tcp.solve(pos, top_down_rotation(np.arctan2(pos[1], pos[0]) - np.pi / 2), q_seed, iters=300)
        R = self.ik_tcp.fk(q)[1]
        q, err_pos, _ = self.ik_tcp.solve(pos, top_down_rotation(np.arctan2(R[1, 2], R[0, 2])), q, iters=300)
        R = self.ik_tcp.fk(q)[1]
        tilt = np.degrees(np.arccos(np.clip(-R[2, 0], -1, 1)))     # site x = approach direction
        return q, err_pos, tilt

    def plan_keyframes(self):
        """7 keyframes (MOVE_PHASES) from the true block pose. None if out of reach."""
        block_pos, _ = self.block_pose()
        az = np.arctan2(block_pos[1], block_pos[0])
        bx, by = block_pos[:2] + GRASP_AIM_LEFT * np.array([-np.sin(az), np.cos(az)])
        place_z = GRASP_Z + PLACE_DZ
        targets = [("PRE_GRASP", HOVER_Z, "open"), ("GRASP", GRASP_Z, "open"), ("CLOSE", GRASP_Z, "closed"),
                   ("LIFT", GRASP_Z + LIFT_DZ, "closed"), ("PLACE", place_z, "closed"),
                   ("RELEASE", place_z, "open"), ("RETREAT", HOVER_Z, "open")]
        keyframes, q = [], self.q_seed4.copy()
        for phase, z, grip in targets:
            pos = np.array([bx, by, z])
            q, err_pos, tilt = self._solve(pos, q)
            max_tilt = MAX_GRASP_TILT_DEG if phase in ("GRASP", "CLOSE", "PLACE", "RELEASE") else 2 * MAX_GRASP_TILT_DEG
            if err_pos > 2e-3 or tilt > max_tilt:
                return None
            keyframes.append(Keyframe(phase, self.full_q(q), GRIPPER[grip], pos))
        return keyframes

    def refine_start(self, q_target, rng, tries=50):
        """A pose near q_target (PRE_GRASP keyframe) for a REFINE sample: inside the limits, the jaw tips
        >= 2 cm above the block top (the jaw tips are ~2 cm below the TCP)."""
        std = rng.uniform(*REFINE_DEG)
        for _ in range(tries):
            q4 = q_target[:4] + np.radians(rng.normal(0, std, 4))
            if np.any(q4 < self.ik_tcp.lo) or np.any(q4 > self.ik_tcp.hi):
                continue
            if self.ik_tcp.fk(q4)[0][2] >= MAT_TOP + 2 * BLOCK_HALF[2] + 0.04:
                return self.full_q(q4)
        return None

    def note_lift(self):
        """Call after LIFT: remembers whether the block came up with the gripper."""
        self.block_lifted = self.block_pose()[0][2] > self.block_start[2] + LIFTED_DZ

    def is_success(self):
        """Lifted during LIFT, and now back on the mat, upright, near where it was."""
        block_pos, _ = self.block_pose()
        upright = self.data.xmat[self.block_body].reshape(3, 3)[2, 2] > np.cos(np.radians(15))
        resting = block_pos[2] < MAT_TOP + BLOCK_HALF[2] + 0.006
        back = np.linalg.norm(block_pos[:2] - self.block_start[:2]) < PUT_BACK_TOL
        return bool(self.block_lifted and upright and resting and back)


def run_expert(env, keyframes, viewer=None):
    for kf in keyframes:
        env.move_to(kf.q, kf.gripper, viewer=viewer)
        if kf.phase == "LIFT":
            env.note_lift()
        if viewer is not None and not viewer.is_running():
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-randomize", action="store_true")
    args = ap.parse_args()

    env = PickEnv(randomize=not args.no_randomize)
    rng = np.random.default_rng(args.seed)
    stats = {"success": 0, "fail": 0, "out_of_reach": 0}

    def run(viewer=None):
        for ep in range(args.episodes):
            env.reset(env.sample_task(rng), rng)
            kfs = env.plan_keyframes()
            if kfs is None:
                stats["out_of_reach"] += 1
                continue
            if not run_expert(env, kfs, viewer):
                return
            ok = env.is_success()
            stats["success" if ok else "fail"] += 1
            print(f"ep {ep}: {'success' if ok else 'FAIL'}  (lifted: {env.block_lifted}, block moved "
                  f"{env.block_moved() * 1000:.1f} mm)")

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
