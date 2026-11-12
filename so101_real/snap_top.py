"""Snap a top-camera click to the object next to it: the white block or the green pad (click_goal_real.py, M key).

A click (yours or the decision model's) only has to say roughly where; the exact point comes from the picture:
  * white block (on the dark mat): a 3 x 3 x 5 cm box is fitted to its white blob - x, y and the turn that make the
    box's outline in the picture match the blob best. The camera sees the block a little from the side, so the
    blob's middle is not the top face's middle; the box fit gets the top face right.
  * green pad: the middle of the green blob, on the table.
Then the arm offset is added (top_arm_offset.json, fit_top_offset.py): where the arm really has to go is ~2 cm
from where the top camera sees the object (the top camera's frame and the arm's disagree by a constant shift, the
same one that makes top + wrist rays "miss" each other by ~2 cm). Your own clicks always included it (you learned
to click up-left of the block); the snapped click does too, so old and new training labels mean the same thing.
Returns the pixel to click so that the click tool's ray (on the selected surface) lands on the offset point.
Nothing is snapped if no object is near enough or the fit is poor (e.g. the arm hides half of the block).
"""
import os
import json

import numpy as np
import cv2

from real_env import MAT_TOP, BLOCK_HALF

BLOCK_TOP = MAT_TOP + 2 * BLOCK_HALF[2]
PAD_Z = 0.003                    # the green pad's top (on the table or the mat: 2-4 mm)
SNAP_MAX_MM = 40.0               # the click this far (on the table) from the object at most
MIN_IOU = 0.6                    # box fit quality needed (outline vs. blob overlap)
FIT_WIDTH = 720                  # pictures are fitted at this width (faster; ~0.9 mm per pixel)
OFFSET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "top_arm_offset.json")


def load_arm_offset():
    """(dx, dy) m from fit_top_offset.py, or None (no file)."""
    try:
        with open(OFFSET_FILE) as f:
            return np.array(json.load(f)["offset_m"], float)
    except (OSError, ValueError, KeyError):
        return None


class TopCam:
    """Pinhole model of the calibrated top camera (MuJoCo camera: looks down its -z, x = image right, y = up)."""

    def __init__(self, pos, R, fovy_deg, w, h):
        self.pos, self.R = np.asarray(pos, float), np.asarray(R, float).reshape(3, 3)
        self.fovy, self.w, self.h = fovy_deg, w, h
        self.f = 0.5 * h / np.tan(np.radians(fovy_deg) / 2)

    @classmethod
    def from_mujoco(cls, model, data, w, h, name="top_camera"):
        cid = model.camera(name).id
        return cls(data.cam_xpos[cid].copy(), data.cam_xmat[cid].copy(), float(model.cam_fovy[cid]), w, h)

    def resized(self, w, h):
        return TopCam(self.pos, self.R, self.fovy, w, h)

    def project(self, P):
        """(N, 3) world points -> (N, 2) pixels."""
        pc = (np.atleast_2d(P) - self.pos) @ self.R
        z = np.where(-pc[:, 2] > 1e-6, -pc[:, 2], 1e-6)
        return np.stack([self.w / 2 + self.f * pc[:, 0] / z - 0.5, self.h / 2 - self.f * pc[:, 1] / z - 0.5], 1)

    def on_plane(self, u, v, z):
        d = self.R @ np.array([(u + 0.5 - self.w / 2) / self.f, -(v + 0.5 - self.h / 2) / self.f, -1.0])
        t = (z - self.pos[2]) / d[2]
        return self.pos + t * d


def _box_corners(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = BLOCK_HALF[0], BLOCK_HALF[1]
    xy = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]]) @ np.array([[c, s], [-s, c]])
    xy = xy + [x, y]
    return np.vstack([np.c_[xy, np.full(4, MAT_TOP)], np.c_[xy, np.full(4, BLOCK_TOP)]])


def masks(img, cam, mat_poly):
    """(white, green) blob masks of the picture. mat_poly: the mat corners in this picture's pixels."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = ((h >= 35) & (h <= 90) & (s >= 60) & (v >= 60)).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    area = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(area, [np.int32(mat_poly)], 255)
    cv2.fillPoly(area, [cv2.convexHull(c) for c in cv2.findContours(green, cv2.RETR_EXTERNAL,
                                                                     cv2.CHAIN_APPROX_SIMPLE)[0]
                        if cv2.contourArea(c) > 200], 255)       # a block on the pad (the pad may be off the mat)
    mat_v = np.median(v[(area > 0) & (green == 0)])
    white = ((s <= 70) & (v >= max(150, mat_v + 60)) & (area > 0)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return white, green


def _blobs(mask, min_area):
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    return [(lab == i, stats[i], cents[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _dist_px(blob, u, v):
    """0 inside the blob, else the pixel distance to its nearest pixel."""
    m = blob[0]
    if 0 <= int(v) < m.shape[0] and 0 <= int(u) < m.shape[1] and m[int(v), int(u)]:
        return 0.0
    ys, xs = np.nonzero(m)
    return float(np.min(np.hypot(xs - u, ys - v)))


def fit_block(blob_mask, cam):
    """Box (x, y, yaw) whose outline matches the white blob best: (x, y, yaw, IoU)."""
    ys, xs = np.nonzero(blob_mask)
    pad = 25
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    x1, y1 = min(xs.max() + pad, blob_mask.shape[1]), min(ys.max() + pad, blob_mask.shape[0])
    crop = blob_mask[y0:y1, x0:x1]
    n_blob = int(crop.sum())
    canvas = np.zeros(crop.shape, np.uint8)
    off = np.array([x0, y0])

    def iou(x, y, yaw):
        pts = cam.project(_box_corners(x, y, yaw)) - off
        canvas[:] = 0
        cv2.fillConvexPoly(canvas, np.int32(np.round(cv2.convexHull(np.float32(pts))[:, 0])), 1)
        inter = int(np.count_nonzero(canvas[crop]))
        return inter / (n_blob + int(canvas.sum()) - inter)

    cx, cy = cam.on_plane(xs.mean(), ys.mean(), MAT_TOP + BLOCK_HALF[2])[:2]
    best = (-1.0, cx, cy, 0.0)
    for yaw in np.radians(np.arange(0, 90, 5)):                     # coarse: 5 deg, 3 mm
        for dx in np.arange(-0.015, 0.0151, 0.003):
            for dy in np.arange(-0.015, 0.0151, 0.003):
                q = iou(cx + dx, cy + dy, yaw)
                if q > best[0]:
                    best = (q, cx + dx, cy + dy, yaw)
    _, bx, by, byaw = best
    for yaw in byaw + np.radians(np.arange(-4, 4.1, 1)):           # fine: 1 deg, 0.5 mm
        for dx in np.arange(-0.003, 0.00301, 0.0005):
            for dy in np.arange(-0.003, 0.00301, 0.0005):
                q = iou(bx + dx, by + dy, yaw)
                if q > best[0]:
                    best = (q, bx + dx, by + dy, yaw)
    q, x, y, yaw = best
    return x, y, yaw % (np.pi / 2), q


def snap(img, u, v, cam, mat_poly, surface_z=BLOCK_TOP, arm_offset=(0.0, 0.0), max_mm=SNAP_MAX_MM):
    """The object nearest the click (u, v) in img (any size; cam and mat_poly for that size).
    Returns a dict: kind ('block' / 'pad' / None), point (3-D, as the top camera sees it), goal (point + arm_offset),
    pixel (to click so that the ray with the surface at surface_z hits goal's x, y; img pixels), dist_mm (click ->
    object, before the offset), iou (block fit), outline (block corners in img pixels), why (if None)."""
    H, W = img.shape[:2]
    s = min(1.0, FIT_WIDTH / W)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    cs = cam.resized(small.shape[1], small.shape[0])
    us, vs = (u + 0.5) * s - 0.5, (v + 0.5) * s - 0.5
    white, green = masks(small, cs, np.asarray(mat_poly, float) * s)
    # the block's expected blob area at the click (its top face alone, at least)
    face = cs.project(_box_corners(*cs.on_plane(us, vs, BLOCK_TOP)[:2], 0.0)[4:])
    min_area = 0.4 * cv2.contourArea(np.float32(face))
    cands = [("block", b, _dist_px(b, us, vs)) for b in _blobs(white, min_area)]
    cands += [("pad", b, _dist_px(b, us, vs)) for b in _blobs(green, 4 * min_area)]
    out = {"kind": None, "point": None, "goal": None, "pixel": None, "dist_mm": None, "iou": None, "outline": None}
    if not cands:
        out["why"] = "no white block or green pad found"
        return out
    # inside a block blob wins (a block on the pad), then inside the pad, then the nearest
    kind, blob, d = min(cands, key=lambda c: (c[2] > 0, c[0] != "block", c[2]))
    if kind == "block":
        x, y, yaw, q = fit_block(blob[0], cs)
        out["iou"] = round(q, 2)
        P = np.array([x, y, BLOCK_TOP])
        out["outline"] = (cs.project(_box_corners(x, y, yaw)) + 0.5) / s - 0.5
        if q < MIN_IOU:
            out["why"] = f"block fit poor (overlap {q:.2f}: partly hidden?)"
            return out
    else:
        ys, xs = np.nonzero(blob[0])
        P = cs.on_plane(xs.mean(), ys.mean(), PAD_Z)
        P[2] = PAD_Z
    click = cs.on_plane(us, vs, P[2])
    out["dist_mm"] = round(float(np.linalg.norm(click[:2] - P[:2])) * 1000, 1)
    if out["dist_mm"] > max_mm:
        out["why"] = f"nearest {kind} is {out['dist_mm']:.0f} mm from the click (max {max_mm:.0f})"
        return out
    goal = P + np.array([arm_offset[0], arm_offset[1], 0.0])
    z = BLOCK_TOP if surface_z is None else surface_z
    px = cs.project(np.array([goal[0], goal[1], z]))[0]
    out.update(kind=kind, point=P, goal=goal, pixel=(px + 0.5) / s - 0.5)
    return out
