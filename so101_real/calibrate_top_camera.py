"""Measure the real top camera's pose and field of view from the mat corners, so the sim renders training
images from exactly that viewpoint.

Click the 4 corners of the mat in the live image, in ANY order (just go around the mat), then click the
robot base (where the arm stands on the table). The mat is square, so its corners alone can't say which edge
the robot is on: every corner order is tried, impossible ones (camera below the table, mirrored) are dropped,
and the one that puts the robot base where you clicked it wins.

The mat size and the robot mount come from ../scene.xml (via real_env.py), so the corners have known positions
in the robot frame; OpenCV solvePnP gives the camera pose. The vertical field of view is found too (the one
with the smallest reprojection error), unless you pass --fovy.

Saves top_camera_calib.json (used by real_env.py from then on) and top_camera_check.jpg: the sim's view from
the calibrated camera blended with the real image - mat, robot base and table edge should line up.

Run in .venv (the arm server is not needed). Move the arm off the mat corners first.

    python so101_real/calibrate_top_camera.py                 # live camera 2, click 4 corners + the base
    python so101_real/calibrate_top_camera.py --image f.jpg --corners 102,210 455,60 590,300 230,440 --base 150,200
"""
import os
import sys
import json
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import (RealSetupEnv, TOP_CAM_CALIB, MAT_HALF, MAT_TOP, MOUNT_SCENE_POS, _DIMS,  # noqa: E402
                      scene_to_robot, quat_from_mat)
from real_robot_policy import crop_to_4_3, TOP_CAM, TOP_RES  # noqa: E402

CORNER_NAMES = ["front-left", "front-right", "back-right", "back-left"]   # as seen from the robot
CHECK_FILE = os.path.join(HERE, "top_camera_check.jpg")


def mat_corners_robot():
    """Mat-top corners in the robot frame, in CORNER_NAMES order (robot +x = into the mat, +y = left)."""
    cx, cy = scene_to_robot([*_DIMS["mat_center"], MOUNT_SCENE_POS[2]])[:2]
    h = MAT_HALF
    return np.array([[cx - h, cy + h, MAT_TOP], [cx - h, cy - h, MAT_TOP],
                     [cx + h, cy - h, MAT_TOP], [cx + h, cy + h, MAT_TOP]])


def capture(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TOP_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TOP_RES[1])
    if not cap.isOpened():
        raise SystemExit(f"camera {index} could not be opened")
    for _ in range(10):  # let auto-exposure settle
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("camera read failed")
    return frame


def click_points(img):
    """4 mat corners (any order around the mat) + the robot base."""
    pts = []
    win = "calibration (u = undo, ENTER = done, q = quit)"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 5:
            pts.append((x, y))

    try:
        cv2.namedWindow(win)
    except cv2.error:
        raise SystemExit("OpenCV has no window support: fix opencv-python, or pass --image, --corners and --base")
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = img.copy()
        for i, p in enumerate(pts):
            color = (0, 0, 255) if i < 4 else (255, 0, 255)
            cv2.circle(view, p, 5, color, -1)
            cv2.putText(view, "ROBOT BASE" if i == 4 else str(i + 1), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts[:4])], len(pts) >= 4, (0, 255, 255), 1)
        if len(pts) < 4:
            msg = f"click mat corner {len(pts) + 1}/4 (any order, go around the mat)"
        elif len(pts) == 4:
            msg = "now click the ROBOT BASE (where the arm stands)"
        else:
            msg = "ENTER to accept, u to undo"
        cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (13, 10) and len(pts) == 5:
            break
        elif k == ord("q"):
            raise SystemExit("cancelled")
    cv2.destroyWindow(win)
    return np.array(pts[:4], float), np.array(pts[4], float)


def fit(world, image_pts, w, h, fovy=None):
    """Best camera for one corner assignment: (fovy_deg, rvec, tvec, rms_px, K), or None if impossible."""
    def pnp(fov):
        f = (h / 2) / np.tan(np.radians(fov) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        ok, rvec, tvec = cv2.solvePnP(world, image_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(world, image_pts, K, None, rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        if (R @ world.T + tvec).T[:, 2].min() <= 0 or (-R.T @ tvec).ravel()[2] < 0.05:
            return None   # mat behind the camera, or camera below the table (mirrored assignment)
        proj, _ = cv2.projectPoints(world, rvec, tvec, K, None)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image_pts) ** 2, axis=1))))
        return fov, rvec, tvec, rms, K

    results = [pnp(fovy)] if fovy is not None else [pnp(fov) for fov in np.arange(20.0, 100.0, 0.25)]
    results = [r for r in results if r is not None]
    return min(results, key=lambda r: r[3]) if results else None


def solve(world, clicked, base_px, w, h, fovy=None):
    """Try all 8 assignments of the clicked corners to the mat corners (4 rotations x 2 directions); keep the
    ones that fit, and choose the one whose robot origin projects closest to the clicked base.
    Without base_px the clicked order is taken as front-left, front-right, back-right, back-left."""
    candidates = []
    orders = [[(s + d * i) % 4 for i in range(4)] for s in range(4) for d in (1, -1)]
    for order in orders if base_px is not None else [[0, 1, 2, 3]]:
        r = fit(world, clicked[order], w, h, fovy)
        if r is None:
            continue
        fov, rvec, tvec, rms, K = r
        base_err = 0.0
        if base_px is not None:
            proj, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, None)
            base_err = float(np.linalg.norm(proj.ravel() - base_px))
        candidates.append((base_err, rms, order, fov, rvec, tvec))
    if not candidates:
        raise SystemExit("could not solve the camera pose - check the clicked corners")
    best_rms = min(c[1] for c in candidates)
    good = [c for c in candidates if c[1] <= max(5.0, 2 * best_rms)]
    return min(good, key=lambda c: c[0])


def opencv_to_mujoco(rvec, tvec):
    """OpenCV world->camera (x right, y down, looks +z) -> MuJoCo camera pos + quat (x right, y up, looks -z)."""
    R = cv2.Rodrigues(rvec)[0]
    pos = (-R.T @ tvec).ravel()
    axes = R.T @ np.diag([1.0, -1.0, -1.0])   # columns: camera x, y, z axes in the robot frame
    return pos, quat_from_mat(axes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=TOP_CAM, help=f"top camera index (default {TOP_CAM})")
    ap.add_argument("--image", help="use this image instead of the live camera")
    ap.add_argument("--corners", nargs=4, metavar="X,Y", help="corner pixels instead of clicking (any order)")
    ap.add_argument("--base", metavar="X,Y", help="robot base pixel (with --corners)")
    ap.add_argument("--fovy", type=float, default=None, help="known vertical field of view (deg); else estimated")
    ap.add_argument("--no-window", action="store_true", help="only save the check image, don't show it")
    args = ap.parse_args()

    frame = cv2.imread(args.image) if args.image else capture(args.cam)
    if frame is None:
        raise SystemExit(f"could not read {args.image}")
    img = crop_to_4_3(frame)   # same crop as real_robot_policy.py
    h, w = img.shape[:2]
    if args.corners:
        clicked = np.array([[float(v) for v in c.split(",")] for c in args.corners])
        base_px = np.array([float(v) for v in args.base.split(",")]) if args.base else None
    else:
        clicked, base_px = click_points(img)

    world = mat_corners_robot()
    base_err, rms, order, fovy, rvec, tvec = solve(world, clicked, base_px, w, h, args.fovy)
    corners_px = clicked[order]
    pos, quat = opencv_to_mujoco(rvec, tvec)
    look = cv2.Rodrigues(rvec)[0].T[:, 2]   # optical axis (OpenCV +z) in the robot frame
    tilt = np.degrees(np.arccos(np.clip(-look[2], -1, 1)))

    for name, p in zip(CORNER_NAMES, corners_px):
        print(f"  {name:11s} = clicked point at ({p[0]:.0f}, {p[1]:.0f})")
    print(f"camera position (robot frame): x {pos[0]:.3f}  y {pos[1]:.3f}  z {pos[2]:.3f} m above the table")
    print(f"tilt from straight down: {tilt:.1f} deg   vertical field of view: {fovy:.1f} deg"
          f"{' (given)' if args.fovy else ' (estimated)'}")
    print(f"corner reprojection error: {rms:.1f} px  (good: < 3 px)")
    if base_px is not None:
        print(f"robot base: predicted {base_err:.0f} px from your click (a rough click is fine; > 60 px: check)")

    with open(TOP_CAM_CALIB, "w") as f:
        json.dump({"pos": pos.round(5).tolist(), "quat": np.round(quat, 6).tolist(), "fovy": round(fovy, 2),
                   "image_size": [w, h], "corners_px": dict(zip(CORNER_NAMES, corners_px.round(1).tolist())),
                   "base_px": None if base_px is None else base_px.round(1).tolist(), "rms_px": round(rms, 2),
                   "fovy_estimated": args.fovy is None, "created": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    print(f"saved {TOP_CAM_CALIB} (real_env.py uses it from now on)")

    # Check image: sim view from the calibrated camera, blended with the real one
    env = RealSetupEnv(randomize=False)
    env.reset(env.sample_task(np.random.default_rng(0)))
    renderer = mujoco.Renderer(env.model, height=h, width=w)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    renderer.update_scene(env.data, camera="top_camera", scene_option=opt)
    sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    check = cv2.addWeighted(img, 0.5, sim, 0.5, 0)
    for name, p in zip(CORNER_NAMES, corners_px.astype(int)):
        cv2.drawMarker(check, tuple(p), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(check, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(check, "real + sim (calibrated): mat edges and robot base should line up", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(CHECK_FILE, check)
    print(f"check image: {CHECK_FILE}")
    if args.no_window:
        return
    try:
        cv2.imshow("calibration check (any key to close)", check)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
