"""Capture wrist-camera calibration data automatically: the top camera finds the cube, the arm goes to a few poses
above it on its own (gripper open), and at each pose the wrist image + the exact joints are saved. fit_wrist_v2
then fits the webcam pose so that the fingers AND the cube land where the real wrist camera shows them.

Needs: arm_server.py running (.venv-lerobot), the top camera aligned (align_top_camera.py), the cube on the mat.
You only place the cube (square to the arm) and press ENTER, --spots times; nothing goes lower than 3 cm above it.

    python so101_real/capture_wrist_calib.py --top-cam 2 --wrist-cam 4
    -> so101_real/captures/wrist_calib_<dd-mm-yyyy-hh-mm-ss>/  spot<n>_pose<k>_wrist.jpg / _top.jpg, poses.jsonl
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import TOP_CAM_CALIB, MAT_TOP, BLOCK_HALF, HOVER_Z, REAL_GRIPPER_OPEN, quat_to_mat  # noqa: E402
from pick_task import PickEnv                                                                   # noqa: E402
from real_arm_client import ArmClient, load_joint_map, sim_to_real                              # noqa: E402
from real_robot_policy import open_camera, crop_to_4_3, fmt, TOP_CAM, WRIST_CAM, TOP_RES, WRIST_RES   # noqa: E402
import align_top_camera as align                                                                # noqa: E402

BLOCK_TOP = MAT_TOP + 2 * BLOCK_HALF[2]
# Poses per cube spot: grasp point (tcp) offset from the cube centre, (along the arm, across, height above the cube top)
POSES = [(0.0, 0.0, HOVER_Z - BLOCK_TOP), (0.0, 0.015, HOVER_Z - BLOCK_TOP), (0.0, -0.015, HOVER_Z - BLOCK_TOP),
         (0.015, 0.0, HOVER_Z - BLOCK_TOP), (-0.015, 0.0, HOVER_Z - BLOCK_TOP), (0.0, 0.0, 0.035)]
MIN_TCP_ABOVE_TOP = 0.03    # the jaw tips are ~2 cm below the tcp: they stay >= 1 cm above the cube
WINDOW = "wrist calibration capture (top | wrist)"


def frame(cap, n_drop=4):
    for _ in range(n_drop):          # drop buffered frames
        cap.grab()
    ok, f = cap.read()
    if not ok:
        raise SystemExit("camera read failed")
    return f


def check_top_alignment(top_img, calib):
    """Back-corner error (px) of the live mat edges vs the calibration, as align_top_camera.py measures it."""
    ref = {k: np.array(v, float) for k, v in calib["corners_px"].items()}
    gray = cv2.GaussianBlur(cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lines = {e: align.find_edge(gray, ref[a], ref[b]) for e, (a, b) in align.EDGES.items()}
    if not all(lines.values()):
        return None
    errs = [np.linalg.norm(align.intersect(lines["back"], lines[s]) - ref[c])
            for s, c in (("left", "back-left"), ("right", "back-right"))]
    return max(errs)


def find_cube(top_img, calib):
    """Cube centre (robot frame) from the top image: the white blob on the mat, its centre ray hit at about
    1 cm below the cube top (the blob includes the sides). Also returns the blob for drawing."""
    h, w = top_img.shape[:2]
    gray = cv2.cvtColor(top_img, cv2.COLOR_BGR2GRAY)
    mat = np.zeros(gray.shape, np.uint8)
    poly = np.array([calib["corners_px"][c] for c in align.CORNERS], np.int32)
    cv2.fillPoly(mat, [poly], 1)
    mat = cv2.erode(mat, np.ones((25, 25), np.uint8))
    thr = max(150, float(np.median(gray[mat > 0])) + 60)
    mask = ((gray > thr) & (mat > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    blobs = [i for i in range(1, n) if 400 < stats[i, cv2.CC_STAT_AREA] < 20000]
    if not blobs:
        return None, None
    i = max(blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    f = (h / 2) / np.tan(np.radians(calib["fovy"]) / 2)
    R, p = quat_to_mat(np.array(calib["quat"])), np.array(calib["pos"])
    u, v = cent[i]
    ray = R @ np.array([(u - w / 2) / f, -(v - h / 2) / f, -1.0])
    z = BLOCK_TOP - 0.01
    cube = p + (z - p[2]) / ray[2] * ray
    cube[2] = MAT_TOP + BLOCK_HALF[2]
    return cube, (lab == i)


def show(top, wrist, text):
    t = cv2.resize(top, (640, 480), interpolation=cv2.INTER_AREA)
    view = np.hstack([t, cv2.resize(wrist, (640, 480))])
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(view, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    try:
        cv2.imshow(WINDOW, view)
        cv2.waitKey(1)
    except cv2.error:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-cam", type=int, default=TOP_CAM)
    ap.add_argument("--wrist-cam", type=int, default=WRIST_CAM)
    ap.add_argument("--spots", type=int, default=3, help="cube positions (6 poses each)")
    ap.add_argument("--speed", type=float, default=15.0, help="joint speed, deg/s")
    ap.add_argument("--confirm-each", action="store_true", help="ENTER before every pose, not only the first")
    args = ap.parse_args()

    with open(TOP_CAM_CALIB) as f:
        calib = json.load(f)
    out = os.path.join(HERE, "captures", f"wrist_calib_{datetime.now():%d-%m-%Y-%H-%M-%S}")
    os.makedirs(out)
    log = open(os.path.join(out, "poses.jsonl"), "a")
    top_cap, wrist_cap = open_camera(args.top_cam, *TOP_RES), open_camera(args.wrist_cam, *WRIST_RES)
    env = PickEnv(randomize=False)
    jm = load_joint_map()
    arm = ArmClient()
    grip = {"gripper": REAL_GRIPPER_OPEN}

    def move(q, what):
        target = {**sim_to_real(q, jm), **grip}
        r = arm.move(target, args.speed)
        if r.get("stopped"):
            raise SystemExit(f"stopped (Ctrl+C on the server) during: {what}")
        return r

    try:
        top = crop_to_4_3(frame(top_cap))
        err = check_top_alignment(top, calib)
        if err is None or err > 12:
            print(f"WARNING: top camera alignment check: {'mat edges not found' if err is None else f'{err:.0f} px off'}"
                  " - run align_top_camera.py first, or the cube positions will be wrong")
            if input("continue anyway? [y/N] ").strip().lower() != "y":
                return
        else:
            print(f"top camera aligned ({err:.0f} px)")
        print(f"saving to {out}")
        input("ENTER = move to the safe pose, then the parked pose (gripper opens) ")
        move(env.q_seed, "safe pose")
        move(env.q_home, "parked pose")
        n_saved = 0
        for spot in range(1, args.spots + 1):
            while True:
                a = input(f"\nspot {spot}/{args.spots}: place the cube 20-30 cm from the base, square to the arm, "
                          "then ENTER (q = finish) ").strip().lower()
                if a == "q":
                    return
                top = crop_to_4_3(frame(top_cap))
                cube, blob = find_cube(top, calib)
                if cube is None:
                    print("  cube not found in the top image (on the mat? arm out of the way?) - try again")
                    continue
                r_xy = np.linalg.norm(cube[:2])
                vis = top.copy()
                vis[blob] = (0, 255, 0)
                show(vis, frame(wrist_cap), f"spot {spot}: cube at x {cube[0]:.3f} y {cube[1]:.3f} ({r_xy * 100:.0f} cm)")
                print(f"  cube found at x {cube[0]:.3f}  y {cube[1]:.3f} m ({r_xy * 100:.0f} cm from the base)")
                if not 0.17 <= r_xy <= 0.33:
                    print("  out of the 18-32 cm range - move it")
                    continue
                if input("  ENTER = go   r = retry : ").strip().lower() != "r":
                    break
            az = np.arctan2(cube[1], cube[0])
            along, across = np.array([np.cos(az), np.sin(az)]), np.array([-np.sin(az), np.cos(az)])
            q = env.q_seed4.copy()
            for k, (da, dc, dz) in enumerate(POSES):
                pos = np.array([*(cube[:2] + da * along + dc * across), BLOCK_TOP + dz])
                q, err_pos, tilt = env._solve(pos, q)
                if err_pos > 2e-3 or pos[2] < BLOCK_TOP + MIN_TCP_ABOVE_TOP - 1e-6:
                    print(f"  pose {k}: unreachable ({err_pos * 1000:.0f} mm) - skipped")
                    continue
                qf = env.full_q(q)
                print(f"  pose {k}: grasp point {da * 100:+.1f} cm along, {dc * 100:+.1f} cm across, "
                      f"{dz * 100:.1f} cm above the cube top -> {fmt(sim_to_real(qf, jm))}")
                if (k == 0 or args.confirm_each) and input("  ENTER = move   s = skip spot : ").strip().lower() == "s":
                    break
                r = move(qf, f"spot {spot} pose {k}")
                time.sleep(0.8)                         # let the arm and the camera exposure settle
                wrist, top = frame(wrist_cap, 6), crop_to_4_3(frame(top_cap))
                name = f"spot{spot}_pose{k}"
                cv2.imwrite(os.path.join(out, f"{name}_wrist.jpg"), wrist, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(out, f"{name}_top.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, 92])
                log.write(json.dumps({"spot": spot, "pose": k, "name": name, "cube_xyz": cube.round(4).tolist(),
                                      "tcp_target": pos.round(4).tolist(), "offset_along_across_up": [da, dc, dz],
                                      "target_sim_deg": np.degrees(qf).round(2).tolist(),
                                      "reached_real_deg": {m: round(v, 2) for m, v in r["joints"].items()},
                                      "max_err_deg": round(r["max_err_deg"], 2)}) + "\n")
                log.flush()
                n_saved += 1
                show(top, wrist, f"spot {spot} pose {k} saved ({n_saved} so far)")
            # back up and out of the top camera's view before the cube is moved
            move(env.full_q(env._solve(np.array([*cube[:2], HOVER_Z + 0.03]), q)[0]), "lift")
            move(env.q_home, "parked pose")
        print(f"\ndone: {n_saved} poses saved in {out}")
    finally:
        log.close()
        if input("\nReturn the arm to its start pose? [Y/n] ").strip().lower() != "n":
            arm.home()
        arm.close()
        top_cap.release()
        wrist_cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
