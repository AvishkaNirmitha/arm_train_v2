"""Point calibration of the wrist camera for click_goal_real.py (hand-eye calibration from clicks).

You click the centre of the block's TOP face in saved wrist images whose joint angles and block position are
known (a capture_wrist_calib.py folder: the top camera measured the block, the arm reported its joints). The
wrist camera pose on the gripper and its fovy are then fitted so those 3-D points project onto your clicks.
Unlike the silhouette fit (fit_wrist_calib.py) this measures the camera's angles directly, and it reports the
error in pixels and in mm (how far the true point is from the clicked ray), plus a leave-one-spot-out check.

Writes wrist_click_calib.json, read only by click_goal_real.py (wrist_camera_calib.json and the VLM sim are not
touched), and keeps your clicks in <captures>/wrist_points.json, so a re-fit doesn't need new clicks.

    python so101_real/fit_wrist_points.py                  # newest captures/wrist_calib_* folder
    python so101_real/fit_wrist_points.py --captures so101_real/captures/wrist_calib_05-11-2026-11-24-04
    python so101_real/fit_wrist_points.py --reclick        # click again

Clicking: left-click = centre of the block's top face (click again to move it), SPACE / ENTER = next,
S = skip (top face not visible), B = back, Q = stop clicking and fit. RED = where the current calibration puts
that point (only a hint: click what you see).
"""
import os
import sys
import glob
import json
import argparse
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import MAT_TOP, BLOCK_HALF, quat_from_mat, quat_to_mat   # noqa: E402
from pick_task import PickEnv                                           # noqa: E402
from real_arm_client import load_joint_map, real_to_sim                 # noqa: E402
from click_goal_real import Kin, WRIST_CLICK_CALIB                      # noqa: E402

BLOCK_TOP = MAT_TOP + 2 * BLOCK_HALF[2]
SHOW = 1.5                    # click window scale (640x480 -> 960x720)
WINDOW = "wrist points: click the CENTRE of the block's TOP face"
# Weak prior (1 sigma) on the change from the current calibration: keeps the fit well posed with few points
PRIOR = np.array([0.02] * 3 + [0.2] * 3 + [10.0])     # m, rad (rotation vector), deg (fovy)
CLICK_PX = 2.0                # assumed click accuracy: pixel residuals are divided by this
SPOT_MAX_OFF = 0.06           # a spot whose clicks put the block farther than this from its record is dropped


class WristFit:
    """Wrist camera pose (gripper body frame) + fovy as x = [dpos (3), rotvec (3), fovy] around the current one."""

    def __init__(self):
        self.env = PickEnv(randomize=False)
        self.kin = Kin(self.env)
        m = self.env.model
        self.cid = m.camera("wrist_camera").id
        self.body = m.cam_bodyid[self.cid]
        self.pos0, self.R0 = m.cam_pos[self.cid].copy(), quat_to_mat(m.cam_quat[self.cid])
        self.x0 = np.array([0, 0, 0, 0, 0, 0, m.cam_fovy[self.cid]], float)

    def gripper_pose(self, q):
        d = self.kin.set(q).d
        return d.xpos[self.body].copy(), d.xmat[self.body].reshape(3, 3).copy()

    def camera(self, x):
        return self.pos0 + x[:3], cv2.Rodrigues(np.asarray(x[3:6], float))[0] @ self.R0, x[6]

    def project(self, x, grip, P, w=640, h=480):
        pos, R, fovy = self.camera(x)
        pg, Rg = grip
        pw, Rw = pg + Rg @ pos, Rg @ R
        pc = Rw.T @ (P - pw)
        f = 0.5 * h / np.tan(np.radians(fovy) / 2)
        return np.array([w / 2 + f * pc[0] / -pc[2] - 0.5, h / 2 - f * pc[1] / -pc[2] - 0.5])

    def ray_miss(self, x, grip, P, uv, w=640, h=480):
        """Distance (m) from the true point P to the ray through the clicked pixel uv."""
        pos, R, fovy = self.camera(x)
        pg, Rg = grip
        pw, Rw = pg + Rg @ pos, Rg @ R
        f = 0.5 * h / np.tan(np.radians(fovy) / 2)
        d = Rw @ np.array([(uv[0] + 0.5 - w / 2) / f, -(uv[1] + 0.5 - h / 2) / f, -1.0])
        d /= np.linalg.norm(d)
        v = P - pw
        return float(np.linalg.norm(v - (v @ d) * d))

    def plane_hit(self, x, grip, uv, z, w=640, h=480):
        """Where the ray through pixel uv meets the plane at height z."""
        pos, R, fovy = self.camera(x)
        pg, Rg = grip
        pw, Rw = pg + Rg @ pos, Rg @ R
        f = 0.5 * h / np.tan(np.radians(fovy) / 2)
        d = Rw @ np.array([(uv[0] + 0.5 - w / 2) / f, -(uv[1] + 0.5 - h / 2) / f, -1.0])
        return pw + (z - pw[2]) / d[2] * d

    def residuals(self, x, items):
        r = [(self.project(x, it["grip"], it["P"]) - it["uv"]) / CLICK_PX for it in items]
        return np.concatenate(r + [(x - self.x0) / PRIOR])

    def fit(self, items, iters=60):
        """Levenberg-Marquardt, numeric Jacobian."""
        x, lam = self.x0.copy(), 1e-3
        eps = np.array([1e-5] * 3 + [1e-5] * 3 + [1e-3])
        r = self.residuals(x, items)
        for _ in range(iters):
            J = np.column_stack([(self.residuals(x + e, items) - r) / e[k]
                                 for k, e in enumerate(np.diag(eps))])
            H, g = J.T @ J, J.T @ r
            while True:
                dx = -np.linalg.solve(H + lam * np.diag(np.diag(H)), g)
                r_new = self.residuals(x + dx, items)
                if r_new @ r_new < r @ r:
                    x, r, lam = x + dx, r_new, max(lam / 3, 1e-7)
                    break
                lam *= 4
                if lam > 1e6:
                    return x
            if np.linalg.norm(dx) < 1e-9:
                break
        return x

    def errors(self, x, items):
        px = np.array([np.linalg.norm(self.project(x, it["grip"], it["P"]) - it["uv"]) for it in items])
        mm = np.array([self.ray_miss(x, it["grip"], it["P"], it["uv"]) * 1000 for it in items])
        return px, mm


def click_points(rows, folder, fitter, clicks):
    """GUI: one top-face-centre click per image. clicks: {name: [u, v] or None (skipped)}, updated in place."""
    print(__doc__.split("Clicking:")[1].strip())
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    st = {"uv": None}
    cv2.setMouseCallback(WINDOW, lambda e, u, v, *_: st.update(uv=[u / SHOW, v / SHOW])
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    i = 0
    while 0 <= i < len(rows):
        row = rows[i]
        img = cv2.imread(os.path.join(folder, row["name"] + "_wrist.jpg"))
        hint = fitter.project(fitter.x0, row["grip"], row["P"])
        st["uv"] = clicks.get(row["name"])
        while True:
            view = cv2.resize(img, None, fx=SHOW, fy=SHOW, interpolation=cv2.INTER_LINEAR)
            cv2.drawMarker(view, tuple(int(c * SHOW) for c in hint), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 1)
            if st["uv"] is not None:
                cv2.drawMarker(view, tuple(int(c * SHOW) for c in st["uv"]), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.rectangle(view, (0, 0), (view.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(view, f"{i + 1}/{len(rows)} {row['name']}: click the CENTRE of the block's TOP face | "
                        "SPACE next  S skip  B back  Q fit", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord(" "), 13, 10) and st["uv"] is not None:
                clicks[row["name"]] = [round(c, 1) for c in st["uv"]]
                i += 1
                break
            if k == ord("s"):
                clicks[row["name"]] = None
                i += 1
                break
            if k == ord("b"):
                i = max(i - 1, 0)
                break
            if k in (ord("q"), 27):
                cv2.destroyWindow(WINDOW)
                return
    cv2.destroyWindow(WINDOW)


def report(title, fitter, x, items):
    px, mm = fitter.errors(x, items)
    print(f"{title}: {px.mean():5.1f} px mean, {np.sqrt((px ** 2).mean()):5.1f} px rms | ray misses the point by "
          f"{mm.mean():5.1f} mm mean, {mm.max():5.1f} mm max")
    return px, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=None, help="capture_wrist_calib.py folder (default: the newest)")
    ap.add_argument("--reclick", action="store_true", help="ignore saved clicks and click again")
    args = ap.parse_args()

    folder = args.captures or max(glob.glob(os.path.join(HERE, "captures", "wrist_calib_*")), default=None)
    if folder is None or not os.path.exists(os.path.join(folder, "poses.jsonl")):
        raise SystemExit("no capture folder with poses.jsonl (run capture_wrist_calib.py, or pass --captures)")
    print(f"captures: {folder}")
    fitter = WristFit()
    jm = load_joint_map()
    rows = [json.loads(line) for line in open(os.path.join(folder, "poses.jsonl")) if line.strip()]
    for row in rows:
        q = real_to_sim(row["reached_real_deg"], jm, fitter.env.q_home)   # wrist_roll: the taped angle
        row["grip"] = fitter.gripper_pose(q)
        row["P"] = np.array([row["cube_xyz"][0], row["cube_xyz"][1], BLOCK_TOP])   # top-face centre

    points_file = os.path.join(folder, "wrist_points.json")
    clicks = {}
    if os.path.exists(points_file) and not args.reclick:
        clicks = json.load(open(points_file))
        print(f"using {len(clicks)} saved clicks from {points_file} (--reclick to click again)")
    if args.reclick or len(clicks) < len(rows):
        click_points(rows, folder, fitter, clicks)
        with open(points_file, "w") as f:
            json.dump(clicks, f, indent=1)
        print(f"clicks saved: {points_file}")

    items = [{"name": r["name"], "spot": r["spot"], "grip": r["grip"], "P": r["P"], "uv": np.array(clicks[r["name"]])}
             for r in rows if clicks.get(r["name"]) is not None]
    # Data check per spot: where the clicks put the block (current calibration, block-top plane) vs where the
    # capture recorded it. A few cm = calibration error (what the fit fixes); far more = the block wasn't there
    print("\nper spot: where your clicks put the block vs where the capture recorded it")
    for s in sorted({it["spot"] for it in items}):
        sp = [it for it in items if it["spot"] == s]
        hits = np.array([fitter.plane_hit(fitter.x0, it["grip"], it["uv"], BLOCK_TOP)[:2] for it in sp])
        off = np.linalg.norm(np.median(hits, axis=0) - sp[0]["P"][:2])
        spread = np.linalg.norm(hits - np.median(hits, axis=0), axis=1).max()
        drop = off > SPOT_MAX_OFF
        print(f"  spot {s}: {off * 100:.1f} cm off (clicks agree within {spread * 100:.1f} cm)"
              + ("  -> DROPPED: the block was not where the capture says" if drop else ""))
        if drop:
            items = [it for it in items if it["spot"] != s]
    if len(items) < 6:
        raise SystemExit(f"only {len(items)} usable clicked images - need at least 6")
    print(f"\n{len(items)} points")
    report("current calibration (wrist_camera_calib.json)", fitter, fitter.x0, items)
    x = fitter.fit(items)
    px, mm = report("point fit                                   ", fitter, x, items)

    # one refit without clear outliers (mis-clicks)
    bad = px > max(3 * np.median(px), 10.0)
    if bad.any() and (~bad).sum() >= 6:
        print("outliers (re-click them with --reclick if they look right):",
              ", ".join(f"{it['name']} {p:.0f} px" for it, p, b in zip(items, px, bad) if b))
        items = [it for it, b in zip(items, bad) if not b]
        x = fitter.fit(items)
        px, mm = report("point fit without the outliers             ", fitter, x, items)
    print("  per image:", "  ".join(f"{it['name']} {p:.1f}px/{e:.1f}mm" for it, p, e in zip(items, px, mm)))

    # honest check: fit without one block position, test on it (spots with the same block position count as one)
    where = [tuple(np.round(it["P"][:2], 3)) for it in items]
    cv = []
    for b in sorted(set(where)):
        train = [it for it, w in zip(items, where) if w != b]
        test = [it for it, w in zip(items, where) if w == b]
        if len(train) >= 6:
            cv.append(fitter.errors(fitter.fit(train), test)[1])
    cv_mm = float(np.concatenate(cv).mean()) if cv else float("nan")
    if cv:
        print(f"leave-one-block-position-out: ray misses a NEW block position by {cv_mm:.1f} mm mean")
    else:
        print(f"only {len(set(where))} block position: no check on new positions possible. Fine for clicks from "
              "the hover pose; for more, capture 2-3 really different spots (capture_wrist_calib.py)")

    pos, R, fovy = fitter.camera(x)
    d_ang = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ fitter.R0.T)[0]))
    print(f"change from wrist_camera_calib.json: lens moved {np.linalg.norm(x[:3]) * 1000:.1f} mm, "
          f"turned {d_ang:.1f} deg, fovy {fitter.x0[6]:.1f} -> {fovy:.1f}")
    out = {"pos": np.round(pos, 5).tolist(), "quat": np.round(quat_from_mat(R), 6).tolist(),
           "fovy": round(float(fovy), 2), "wrist_roll_deg": round(float(np.degrees(fitter.env.roll)), 2),
           "rms_px": round(float(np.sqrt((px ** 2).mean())), 2), "ray_mm": round(float(mm.mean()), 2),
           "cv_ray_mm": round(cv_mm, 2), "n_points": len(items), "data": folder,
           "created": datetime.now().isoformat(timespec="seconds")}
    with open(WRIST_CLICK_CALIB, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {WRIST_CLICK_CALIB} (click_goal_real.py uses it from now on)")


if __name__ == "__main__":
    main()
