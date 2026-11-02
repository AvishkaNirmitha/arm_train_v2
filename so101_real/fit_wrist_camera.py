"""Measure where the real wrist webcam sits on the gripper (position, tilt, field of view), so the sim's wrist
camera sees what the real one sees.

Uses captures from capture_ui.py (top + wrist image + joint angles). Per capture:
  * top image   -> the white block's position and yaw on the mat (calibrated top camera, top_camera_calib.json)
  * joint angles -> where the gripper is (sim kinematics, real_joint_map.json)
  * wrist image -> the block's outline
The webcam pose is adjusted until the block outline predicted from the sim matches the real one in all captures.
If the arm hides the block from the top camera, the block pose of the previous capture (by save time) is used:
don't move the block between those captures.

Captures to take (arm_server.py running, both crop areas removed = full frames, top camera at 1920x1080):
  1. block in the middle area of the mat (>= 12 cm from the mat edges)
  2. one capture with the arm away from the block (the top camera must see it)
  3. without moving the block: 3-4 captures with the arm at different poses / heights (8-20 cm) above it,
     the WHOLE block visible in the wrist camera
  4. repeat 1-3 for 2-3 block positions (10-15 captures)

Saves wrist_camera_calib.json (used by real_env.py from then on) and check images in <captures>/wrist_fit_check/:
the real wrist image with the predicted block outline, before (red) and after (green) the fit.

Run in .venv (the arm server is not needed):

    python so101_real/fit_wrist_camera.py                        # all captures in so101_real/captures
    python so101_real/fit_wrist_camera.py --captures DIR --only 29-10-2026
"""
import os
import sys
import glob
import json
import argparse
from datetime import datetime

import cv2
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import (build_real_model, ARM_JOINTS, MAT_TOP, MAT_HALF, BLOCK_HALF, MOUNT_SCENE_POS, _DIMS,  # noqa: E402
                      scene_to_robot, top_camera_pose, default_webcam_pose, WRIST_CAM_CALIB, quat_from_mat,
                      quat_to_mat, TOP_CAM_CALIB)
from real_arm_client import load_joint_map, real_to_sim   # noqa: E402

BLOCK_TOP = MAT_TOP + 2 * BLOCK_HALF[2]
MAT_INNER = 0.015          # look for the block this far inside the mat edges


# ── Image geometry ──────────────────────────────────────────────────────────
def crop_homography(meta, cam):
    """Full camera frame pixel -> pixel of the saved image (capture_ui.py crop areas)."""
    c = meta["crop"][cam]
    if c is None:
        return np.eye(3)
    pts = np.float32(c["corners"])
    w, h = meta["image_size"][cam]
    if len(pts) == 4 and c.get("saved_as") in ("perspective_corrected", "rotated_upright"):
        return cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    x, y, _, _ = cv2.boundingRect(pts.round().astype(np.int32))   # masked bounding-box cut
    return np.array([[1.0, 0, -x], [0, 1.0, -y], [0, 0, 1.0]])


def crop_4_3_offset(fw, fh):
    """Offset and size of the centre 4:3 crop the sim cameras correspond to (real_robot_policy.crop_to_4_3)."""
    if fw * 3 > fh * 4:
        nw = fh * 4 // 3
        return (fw - nw) // 2, 0, nw, fh
    if fw * 3 < fh * 4:
        nh = fw * 3 // 4
        return 0, (fh - nh) // 2, fw, nh
    return 0, 0, fw, fh


class PinholeView:
    """A MuJoCo camera (world pos, R columns = cam x/y/z, fovy) seen through a capture's crop."""

    def __init__(self, meta, cam):
        fw, fh = meta["frame_size"][cam]
        self.x0, self.y0, self.w, self.h = crop_4_3_offset(fw, fh)
        self.H = crop_homography(meta, cam)

    def project(self, pts, pos, R, fovy):
        """World points -> saved-image pixels; None if any point is behind the camera."""
        pc = (np.atleast_2d(pts) - pos) @ R
        if np.any(pc[:, 2] > -1e-3):
            return None
        f = (self.h / 2) / np.tan(np.radians(fovy) / 2)
        uv = np.column_stack([self.w / 2 + f * pc[:, 0] / -pc[:, 2] + self.x0,
                              self.h / 2 - f * pc[:, 1] / -pc[:, 2] + self.y0])
        return cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float64), self.H).reshape(-1, 2)

    def backproject(self, px, pos, R, fovy, z):
        """Saved-image pixels -> points on the horizontal plane at height z."""
        uv = cv2.perspectiveTransform(np.float64(px).reshape(-1, 1, 2), np.linalg.inv(self.H)).reshape(-1, 2)
        f = (self.h / 2) / np.tan(np.radians(fovy) / 2)
        out = []
        for u, v in uv:
            ray = R @ np.array([(u - self.x0 - self.w / 2) / f, -(v - self.y0 - self.h / 2) / f, -1.0])
            out.append(pos + (z - pos[2]) / ray[2] * ray)
        return np.array(out)


# ── Detection ───────────────────────────────────────────────────────────────
def bright_blobs(gray, mask=None):
    """Connected bright regions (white block on the dark mat): Otsu threshold, at least 120."""
    vals = gray[mask > 0] if mask is not None else gray.ravel()
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = ((gray > max(t, 120)) * 255).astype(np.uint8)
    if mask is not None:
        b &= mask
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    return [(lab == i, stats[i]) for i in range(1, n)]


def top_face(gray, blob):
    """The brightest part of a block blob: its top face (the lit side faces are a little darker)."""
    level = np.percentile(gray[blob], 95)
    face = (blob & (gray > level - 35)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(face)
    if n < 2:
        return None
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(lab == i)
    return np.column_stack([xs, ys]).astype(np.float32)


def block_from_top(img, top_view, cam):
    """Block centre xy (robot frame) + yaw (mod 90 deg) from the top image, or (None, reason)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = MAT_HALF - MAT_INNER
    cx, cy = scene_to_robot([*_DIMS["mat_center"], MOUNT_SCENE_POS[2]])[:2]
    mat = [[cx - h, cy + h, MAT_TOP], [cx - h, cy - h, MAT_TOP], [cx + h, cy - h, MAT_TOP], [cx + h, cy + h, MAT_TOP]]
    poly = top_view.project(np.array(mat), *cam)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [poly.round().astype(np.int32)], 255)
    best = None
    for blob, st in bright_blobs(gray, mask):
        face = top_face(gray, blob) if st[cv2.CC_STAT_AREA] >= 20 else None
        if face is None:
            continue
        rect = cv2.minAreaRect(face)
        corners = top_view.backproject(cv2.boxPoints(rect), *cam, z=BLOCK_TOP)
        sides = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        err = np.abs(sides - 2 * BLOCK_HALF[0]).max()    # each side should be 3 cm
        if err < 0.008 and (best is None or err < best[0]):
            best = (err, corners)
    if best is None:
        return None, "no 3 x 3 cm white block found on the mat"
    corners = best[1]
    edge = corners[1] - corners[0]
    yaw = np.arctan2(edge[1], edge[0]) % (np.pi / 2)
    return (corners[:, :2].mean(axis=0), yaw), f"side error {best[0] * 1000:.0f} mm"


def block_outline_wrist(img):
    """Convex outline (N x 2 px) of the block's top face in the wrist image, or (None, reason).
    Bright regions touching the image border (the table beyond the mat, a cut-off block) are ignored."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    inside = [(blob, st) for blob, st in bright_blobs(gray)
              if st[0] > 2 and st[1] > 2 and st[0] + st[2] < W - 2 and st[1] + st[3] < H - 2]
    if not inside:
        return None, "no whole block in view (cut by the image border?)"
    blob, st = max(inside, key=lambda b: b[1][cv2.CC_STAT_AREA])
    if st[cv2.CC_STAT_AREA] < 0.01 * W * H:
        return None, "block too small in the wrist image"
    face = top_face(gray, blob)
    if face is None or len(face) < 0.005 * W * H:
        return None, "top face not found"
    return cv2.convexHull(face).reshape(-1, 2), "ok"


def top_face_corners(xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = BLOCK_HALF[0], BLOCK_HALF[1]
    return np.array([[xy[0] + c * dx - s * dy, xy[1] + s * dx + c * dy, BLOCK_TOP]
                     for dx, dy in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))])


def shape_features(poly, width):
    """Centroid, size and second moments of a polygon, scaled by the image width."""
    m = cv2.moments(np.float32(poly).reshape(-1, 1, 2))
    a = max(m["m00"], 1e-6)
    return np.array([m["m10"] / a, m["m01"] / a, np.sqrt(a), np.sqrt(max(m["mu20"] / a, 0)),
                     np.sqrt(max(m["mu02"] / a, 0)), m["mu11"] / a / width]) / width


# ── Fit ─────────────────────────────────────────────────────────────────────
def rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


class WebcamFit:
    """p = [dx, dy, dz (cm), rx, ry, rz (deg, camera frame), dfovy (deg)] relative to a start pose."""

    def __init__(self, samples, start):
        self.samples = samples
        self.pos0, self.R0, self.fovy0 = start

    def pose(self, p):
        return self.pos0 + p[:3] / 100, self.R0 @ rodrigues(np.radians(p[3:6])), self.fovy0 + p[6]

    def predict(self, s, p):
        pos_b, R_b, fovy = self.pose(p)
        pos = s["g_pos"] + s["g_R"] @ pos_b
        R = s["g_R"] @ R_b
        px = s["wrist_view"].project(top_face_corners(*s["block"]), pos, R, fovy)
        return None if px is None else cv2.convexHull(np.float32(px)).reshape(-1, 2)

    def residuals(self, p):
        r = []
        for s in self.samples:
            poly = self.predict(s, p)
            f = np.full(6, 1.0) if poly is None else shape_features(poly, s["width"])
            r.extend(f - s["features"])
        # pull towards the start pose (~ the size of a 13 px outline error for 3 cm / 30 deg / 10 deg off):
        # keeps the fit sensible with few captures or a wrong one
        r.extend(0.02 * np.concatenate([p[:3] / 3, p[3:6] / 30, p[6:] / 10]))
        return np.array(r)

    def solve(self, p, iters=100):
        lam = 1e-2
        cost = np.sum(self.residuals(p) ** 2)
        for _ in range(iters):
            r = self.residuals(p)
            J = np.zeros((len(r), len(p)))
            for i in range(len(p)):
                dp = np.zeros(len(p))
                dp[i] = 0.05
                J[:, i] = (self.residuals(p + dp) - r) / 0.05
            A = J.T @ J
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -J.T @ r)
            new = np.sum(self.residuals(p + step) ** 2)
            if new < cost:
                p, cost, lam = p + step, new, lam / 3
                if np.abs(step).max() < 1e-4:
                    break
            else:
                lam *= 4
                if lam > 1e8:
                    break
        return p, cost

    def per_sample_px(self, p):
        """Outline mismatch per capture: centroid + size error in pixels."""
        out = []
        for s in self.samples:
            poly = self.predict(s, p)
            if poly is None:
                out.append(np.inf)
                continue
            d = (shape_features(poly, s["width"]) - s["features"])[:3] * s["width"]
            out.append(float(np.linalg.norm(d)))
        return np.array(out)


# ── Main ────────────────────────────────────────────────────────────────────
def load_samples(files, model, data, joint_map, top_cam):
    adr = [model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS]
    gid = model.body("gripper").id
    samples, skipped, block = [], [], None
    for f in files:
        meta = json.load(open(f))
        name = meta.get("name", os.path.basename(f))
        folder = os.path.dirname(f)
        top = cv2.imread(os.path.join(folder, meta["images"]["top"]))
        wrist = cv2.imread(os.path.join(folder, meta["images"]["wrist"]))
        found, why = block_from_top(top, PinholeView(meta, "top"), top_cam)
        if found is not None:
            block = found
            print(f"  {name}: block at x {block[0][0]:.3f} y {block[0][1]:.3f} m, yaw {np.degrees(block[1]):.0f} deg ({why})")
        if not meta.get("joints_deg"):
            skipped.append((name, "no joint angles (arm not connected when saved)"))
            continue
        outline, why = block_outline_wrist(wrist)
        if outline is None:
            skipped.append((name, f"wrist: {why}"))
            continue
        if block is None:
            skipped.append((name, "top camera hasn't seen the block yet"))
            continue
        data.qpos[adr] = real_to_sim(meta["joints_deg"], joint_map)   # wrist_roll stays 0 (as in the sim)
        mujoco.mj_kinematics(model, data)
        samples.append({"name": name, "meta": meta, "wrist_img": wrist, "block": block,
                        "block_seen_now": found is not None, "outline": outline,
                        "features": shape_features(outline, wrist.shape[1]), "width": wrist.shape[1],
                        "wrist_view": PinholeView(meta, "wrist"),
                        "g_pos": data.xpos[gid].copy(), "g_R": data.xmat[gid].reshape(3, 3).copy()})
    return samples, skipped


def save_checks(fit, p0, p, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tiles = []
    for s, e0, e in zip(fit.samples, fit.per_sample_px(p0), fit.per_sample_px(p)):
        img = s["wrist_img"].copy()
        cv2.polylines(img, [s["outline"].round().astype(np.int32)], True, (255, 255, 255), 1)
        for q, color in ((p0, (0, 0, 255)), (p, (0, 255, 0))):
            poly = fit.predict(s, q)
            if poly is not None:
                cv2.polylines(img, [poly.round().astype(np.int32)], True, color, 2)
        cv2.putText(img, f"{s['name']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, f"before {e0:.0f} px  after {e:.0f} px", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f"{s['name']}.jpg"), img)
        tiles.append(cv2.resize(img, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(out_dir, "all.jpg"), grid)
    return os.path.join(out_dir, "all.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", default=os.path.join(HERE, "captures"))
    ap.add_argument("--only", default="", help="use only captures whose file name contains this text")
    ap.add_argument("--out", default=WRIST_CAM_CALIB, help="where to save the result")
    ap.add_argument("--no-save", action="store_true", help="fit and show, don't write the calibration")
    args = ap.parse_args()

    if not os.path.exists(TOP_CAM_CALIB):
        raise SystemExit("needs top_camera_calib.json (calibrate_top_camera.py) to locate the block")
    files = [f for f in glob.glob(os.path.join(args.captures, "*.json"))
             if not f.endswith("crop_areas.json") and args.only in os.path.basename(f)]
    files.sort(key=lambda f: json.load(open(f)).get("saved_at", ""))
    print(f"{len(files)} captures in {args.captures}")

    model = build_real_model(as_built=True)
    data = mujoco.MjData(model)
    tc = top_camera_pose()
    top_cam = (tc["pos"], quat_to_mat(tc["quat"]), tc["fovy"])
    samples, skipped = load_samples(files, model, data, load_joint_map(), top_cam)
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    if len(samples) < 4:
        raise SystemExit(f"only {len(samples)} usable captures - need at least 4 (better 10+)")

    start = default_webcam_pose()
    fit = WebcamFit(samples, start)
    p0 = np.zeros(7)
    # The photo estimate may be off a lot: also start from other tilts and the image turned 180 deg
    starts = [np.array([0, 0, 0, rx, 0, rz, 0], float) for rx in (0, -15, 15) for rz in (0, 180)]
    p, cost = min((fit.solve(s, iters=40) for s in starts), key=lambda r: r[1])
    p, cost = fit.solve(p)
    err = fit.per_sample_px(p)
    bad = err > max(3 * np.median(err), 15)
    if bad.any() and (~bad).sum() >= 4:
        print("  left out (outline far off - block moved, or detection wrong?):",
              [s["name"] for s, b in zip(samples, bad) if b])
        fit = WebcamFit([s for s, b in zip(samples, bad) if not b], start)
        p, cost = fit.solve(p)
        err = fit.per_sample_px(p)

    pos, R, fovy = fit.pose(p)
    pos0, R0, fovy0 = start
    view0, view = -R0[:, 2], -R[:, 2]
    turn = np.degrees(np.arccos(np.clip(view0 @ view, -1, 1)))
    print(f"\nused {len(fit.samples)} captures; outline error per capture: median {np.median(err):.1f} px, "
          f"worst {err.max():.1f} px (before the fit: median {np.median(fit.per_sample_px(p0)):.1f} px)")
    print(f"webcam lens moved {np.linalg.norm(pos - pos0) * 1000:.1f} mm, view direction turned {turn:.1f} deg, "
          f"field of view {fovy0:.1f} -> {fovy:.1f} deg (from the photo estimate)")
    check = save_checks(fit, p0, p, os.path.join(args.captures, "wrist_fit_check"))
    print(f"check images: {check}  (white = real outline, red = before, green = after)")
    if args.no_save:
        return
    with open(args.out, "w") as f:
        json.dump({"pos": pos.round(5).tolist(), "quat": np.round(quat_from_mat(R), 6).tolist(),
                   "fovy": round(float(fovy), 2), "frame": "gripper body; MuJoCo camera axes",
                   "captures": [s["name"] for s in fit.samples], "median_px": round(float(np.median(err)), 1),
                   "created": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"saved {args.out} (real_env.py uses it from now on)")


if __name__ == "__main__":
    main()
