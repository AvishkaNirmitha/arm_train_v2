"""Fit the wrist webcam from capture_wrist_calib.py data: per pose the real wrist image, the joints the arm reached
and the cube position from the top camera. The sim is rendered at those joints and the fit makes two things match
in every image at once:
    robot : the black fingers / gripper (dark pixels connected to the bottom edge)
    bright: the white cube and the white table beyond the mat
The fingers alone can't tell "further back with a narrower lens" from "closer with a wider one"; the cube and the
mat edge further away can.

Free parameters: lens pose in the gripper body frame (6) + fovy, the taped wrist_roll angle, the open jaw angle,
and a small cube correction (x, y, yaw). Writes wrist_camera_calib.json (real_env.py uses it from then on) and
wrist_camera_calib_check.jpg (real image with the fitted sim outlines + the sim view).

    python so101_real/fit_wrist_calib.py so101_real/captures/wrist_calib_<date>   # ~15-25 min, prints progress
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pick_task import PickEnv                                                        # noqa: E402
from real_env import (WRIST_CAM_CALIB, MAT_TOP, BLOCK_HALF, quat_from_mat, quat_to_mat,  # noqa: E402
                      AS_BUILT_WRIST_ROLL, AS_BUILT_GRIPPER_OPEN)
from real_arm_client import load_joint_map, real_to_sim                              # noqa: E402
from real_dataset import make_scene_option                                           # noqa: E402

ROBOT_THR = 28      # gray < this and connected to the bottom edge = robot (the arm's shadow on the mat is ~35-50)
BRIGHT_THR = 150    # gray > this = white cube / table
CHECK_FILE = os.path.join(HERE, "wrist_camera_calib_check.jpg")


def real_classes(img, w, h):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    m = cv2.morphologyEx((g < ROBOT_THR).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(m)
    robot = np.isin(lab, list(set(np.unique(lab[-3:, :])) - {0}))
    bright = cv2.morphologyEx((g > BRIGHT_THR).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    small = lambda x: cv2.resize(x.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    return small(robot), small(bright)


def nelder_mead(f, x0, step, iters, report=None, patience=60, tol=1e-3):
    """Stops early when the best loss hasn't improved by tol for patience iterations."""
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    last_best, last_it = min(vals), 0
    for it in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if report and it % 50 == 0:
            report(it, vals[0])
        if vals[0] < last_best - tol:
            last_best, last_it = vals[0], it
        elif it - last_it >= patience:
            if report:
                report(it, vals[0], done=True)
            break
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return vals[i], pts[i]


class Fitter:
    def __init__(self, rows, folder, w, h):
        self.env = PickEnv(randomize=False)
        env = self.env
        env.reset(env.sample_task(np.random.default_rng(0)))
        self.m, self.d = env.model, env.data
        m = self.m
        self.cam = m.camera("wrist_camera").id
        self.body = m.cam_bodyid[self.cam]
        root = m.body_rootid[m.body("gripper").id]
        self.robot_geom = np.array([m.body_rootid[m.geom_bodyid[g]] == root for g in range(m.ngeom)])
        self.bright_geom = np.zeros(m.ngeom, bool)
        self.bright_geom[[m.geom("white_block_geom").id, m.geom("table_top").id]] = True
        for g in ("webcam", "webcam_mount"):      # built at the old lens pose: hidden while the lens moves
            m.geom_group[m.geom(g).id] = 3
            self.robot_geom[m.geom(g).id] = False
        self.roll_qadr = env.roll_qadr
        self.w, self.h = w, h
        self.r = mujoco.Renderer(m, h, w)
        self.r.enable_segmentation_rendering()
        self.opt = make_scene_option()
        jm = load_joint_map()
        self.items = []
        for row in rows:
            img = cv2.imread(os.path.join(folder, row["name"] + "_wrist.jpg"))
            robot, bright = real_classes(img, w, h)
            q = real_to_sim(row["reached_real_deg"], jm, env.q_home)
            self.items.append({"q": q, "cube": np.array(row["cube_xyz"]), "robot": robot, "bright": bright,
                               "robot_b": cv2.GaussianBlur(robot, (0, 0), 2), "bright_b": cv2.GaussianBlur(bright, (0, 0), 2)})
        # start from the lens the model has now (wrist_camera_calib.json if there is one)
        self.pos0, self.R0 = m.cam_pos[self.cam].copy(), quat_to_mat(m.cam_quat[self.cam].copy())
        self.fovy0 = float(m.cam_fovy[self.cam])

    # x: lens dpos (3, body frame, m), rotvec (3), fovy, roll (deg), jaw (deg), cube dx, dy (m), cube yaw (deg)
    def apply(self, x, item):
        m, d = self.m, self.d
        m.cam_pos[self.cam] = self.pos0 + x[0:3]
        m.cam_quat[self.cam] = quat_from_mat(cv2.Rodrigues(np.asarray(x[3:6], float))[0] @ self.R0)
        m.cam_fovy[self.cam] = x[6]
        q = item["q"].copy()
        q[4] = np.radians(x[7])
        d.qpos[self.env.arm_qadr] = q
        d.qpos[self.env.grip_qadr] = np.radians(x[8])
        c = item["cube"]
        az = np.arctan2(c[1], c[0]) + np.radians(x[11])
        qa = self.env.block_qadr
        d.qpos[qa:qa + 3] = [c[0] + x[9], c[1] + x[10], MAT_TOP + BLOCK_HALF[2]]
        d.qpos[qa + 3:qa + 7] = [np.cos(az / 2), 0, 0, np.sin(az / 2)]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)

    def render(self, x, item):
        self.apply(x, item)
        self.r.update_scene(self.d, camera="wrist_camera", scene_option=self.opt)
        seg = self.r.render()
        ids = np.clip(seg[..., 0], 0, None)
        geom = (seg[..., 0] >= 0) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        return (geom & self.robot_geom[ids]).astype(np.float32), (geom & self.bright_geom[ids]).astype(np.float32)

    def loss(self, x, items=None, bright=True):
        """bright=False: the robot (fingers / gripper) only - stage 1, finding the lens near the gripper."""
        total = 0.0
        items = items or self.items
        for it in items:
            sr, sb = self.render(x, it)
            pairs = ((sr, it["robot"], it["robot_b"]), (sb, it["bright"], it["bright_b"]))[:2 if bright else 1]
            for s, real, real_b in pairs:
                union = np.maximum(s, real).sum()
                total += 1 - (s * real).sum() / max(union, 1) + 2 * np.mean((cv2.GaussianBlur(s, (0, 0), 2) - real_b) ** 2)
        return total / ((2 if bright else 1) * len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="captures/wrist_calib_<date> from capture_wrist_calib.py")
    ap.add_argument("--coarse", type=int, default=5000, help="random starting guesses (stage 1, fingers only)")
    ap.add_argument("--iters", type=int, default=1200, help="Nelder-Mead iterations per refinement")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.folder, "poses.jsonl"))]
    seen, unique = set(), []
    for r in rows:                     # the same arm pose + cube again adds nothing but time
        key = (tuple(r["cube_xyz"]), tuple(r["offset_along_across_up"]))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f"{len(rows)} poses, {len(unique)} distinct", flush=True)
    fit = Fitter(unique, args.folder, 128, 96)
    coarse_items = fit.items[::2]
    t0 = time.time()
    x0 = np.array([0, 0, 0, 0, 0, 0, fit.fovy0, np.degrees(AS_BUILT_WRIST_ROLL), np.degrees(AS_BUILT_GRIPPER_OPEN), 0, 0, 0])
    rng = np.random.default_rng(0)
    # Stage 1: the lens from the fingers / gripper alone, searched widely (a re-mounted webcam can sit cm away)
    robot_only = lambda x, items=None: fit.loss(x, items or coarse_items, bright=False)
    best = (robot_only(x0), x0)
    print(f"stage 1 (fingers only), start = current sim webcam: loss {best[0]:.3f}", flush=True)
    for i in range(args.coarse):
        x = x0 + np.concatenate([rng.uniform(-0.05, 0.05, 3), rng.normal(0, 0.5, 3), [rng.uniform(-12, 18)],
                                 [rng.uniform(-10, 10)], [rng.uniform(-8, 8)], [0, 0, 0]])
        l = robot_only(x)
        if l < best[0]:
            best = (l, x)
        if (i + 1) % 250 == 0:
            print(f"stage 1 search {i + 1}/{args.coarse}: best loss {best[0]:.3f}  ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    step = [0.01, 0.01, 0.01, 0.08, 0.08, 0.08, 3, 4, 3, 0.004, 0.004, 5]
    stage1 = []
    for k in range(2):
        start = best[1] if k == 0 else stage1[0][1] + rng.normal(0, 0.3, 12) * step
        fun, xb = nelder_mead(lambda x: fit.loss(x, bright=False), start, step, args.iters,
                              report=lambda it, v, done=False: print(
                                  f"stage 1 refine {k + 1}/2 step {it}: loss {v:.3f}{'  (done)' if done else ''}"
                                  f"  ({(time.time() - t0) / 60:.1f} min)", flush=True))
        stage1.append((fun, xb))
        stage1.sort(key=lambda c: c[0])
    best = stage1[0]
    print(f"stage 1 result: finger loss {best[0]:.3f}", flush=True)
    # Stage 2: fingers + cube + table together
    cands = []
    for k in range(3):
        start = best[1] if k == 0 else cands[0][1] + rng.normal(0, 0.3, 12) * step
        fun, xb = nelder_mead(fit.loss, start, step, args.iters,
                              report=lambda it, v, done=False: print(
                                  f"stage 2 refine {k + 1}/3 step {it}: loss {v:.3f}{'  (done)' if done else ''}"
                                  f"  ({(time.time() - t0) / 60:.1f} min)", flush=True))
        cands.append((fun, xb))
        cands.sort(key=lambda c: c[0])
    loss, x = cands[0]
    fit.apply(x, fit.items[0])
    m = fit.m
    pos, quat, fovy = m.cam_pos[fit.cam].copy(), m.cam_quat[fit.cam].copy(), float(m.cam_fovy[fit.cam])
    ious = []
    for it in fit.items:
        sr, sb = fit.render(x, it)
        ious.append([(s * r).sum() / max(np.maximum(s, r).sum(), 1) for s, r in ((sr, it["robot"]), (sb, it["bright"]))])
    ious = np.array(ious)
    print(f"\nfit: loss {loss:.3f} | overlap robot {ious[:, 0].mean() * 100:.0f} %, cube+table {ious[:, 1].mean() * 100:.0f} %")
    print(f"lens pos (gripper body) {pos.round(4)}  quat {quat.round(4)}  fovy {fovy:.1f}")
    print(f"wrist_roll {x[7]:.1f} deg  jaw open {x[8]:.1f} deg  cube correction {x[9] * 1000:+.0f} / {x[10] * 1000:+.0f} mm, "
          f"yaw {x[11]:+.0f} deg")
    with open(WRIST_CAM_CALIB, "w") as f:
        json.dump({"pos": pos.round(5).tolist(), "quat": quat.round(6).tolist(), "fovy": round(fovy, 2),
                   "wrist_roll_deg": round(float(x[7]), 2), "gripper_open_deg": round(float(x[8]), 2),
                   "overlap_robot": round(float(ious[:, 0].mean()), 3), "overlap_bright": round(float(ious[:, 1].mean()), 3),
                   "data": os.path.abspath(args.folder), "created": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    print(f"saved {WRIST_CAM_CALIB}")

    # check image: real + fitted outlines (red robot, cyan cube/table) | sim render, for 3 poses
    W, H = 640, 480
    big = Fitter(unique[:1], args.folder, W, H)
    rgb = mujoco.Renderer(big.m, H, W)
    panels = []
    for row in unique[:: max(1, len(unique) // 3)][:3]:
        it = {"q": real_to_sim(row["reached_real_deg"], load_joint_map(), big.env.q_home), "cube": np.array(row["cube_xyz"])}
        sr, sb = big.render(x, it)
        real = cv2.imread(os.path.join(args.folder, row["name"] + "_wrist.jpg"))
        for mask, col in ((sr, (0, 0, 255)), (sb, (255, 255, 0))):
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(real, cnts, -1, col, 2)
        big.apply(x, it)
        rgb.update_scene(big.d, camera="wrist_camera", scene_option=big.opt)
        sim = cv2.cvtColor(rgb.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(real, f"REAL {row['name']} + fitted sim outlines", (8, 24), 0, 0.6, (0, 255, 255), 2)
        cv2.putText(sim, "SIM wrist view (fitted)", (8, 24), 0, 0.6, (0, 255, 255), 2)
        panels.append(np.hstack([real, sim]))
    cv2.imwrite(CHECK_FILE, np.vstack(panels))
    print(f"check image: {CHECK_FILE}")


if __name__ == "__main__":
    main()
