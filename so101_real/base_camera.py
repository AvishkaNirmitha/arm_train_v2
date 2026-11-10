"""The base camera: a webcam on the arm's base that TURNS WITH shoulder_pan (it sits on the "shoulder" body), looking
forward along the arm (it sees the jaws from behind / below and the mat in front of them).

Its pose is stored relative to the shoulder body (base_camera_calib.json, from calib_base_camera.py), so its place in
the world comes from the shoulder_pan angle alone. Camera axes are OpenCV's: x = image right, y = image down,
z = forward (the viewing direction). Pinhole, no distortion (the same camera model as the top camera, which fits a
pinhole to ~2 px).
"""
import os
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_CLICK_CALIB = os.path.join(HERE, "base_camera_calib.json")
BODY = "shoulder"                          # the body the camera is fixed to (turns with shoulder_pan)
# Tip of the FIXED jaw in the gripper body frame: the sim mesh's points within 3 mm of its far end (wrist_roll_follower)
FIXED_TIP = np.array([-0.0102, -0.0002, -0.1033])
TOP_FOVY = 58.5                            # the top camera's (same model, same 1920x1080 mode cropped to 4:3)


def focal_px(fovy_deg, h):
    return 0.5 * h / np.tan(np.radians(fovy_deg) / 2)


def fixed_tip(kin):
    """World position of the fixed jaw's tip for the pose kin was last set to."""
    b = kin.m.body("gripper").id
    return kin.d.xpos[b] + kin.d.xmat[b].reshape(3, 3) @ FIXED_TIP


def body_pose(kin):
    b = kin.m.body(BODY).id
    return kin.d.xpos[b].copy(), kin.d.xmat[b].reshape(3, 3).copy()


class BaseCam:
    """R (camera -> shoulder body rotation), t (lens in the shoulder body frame, m), fovy (deg), image w x h."""

    def __init__(self, R, t, fovy, w, h, rms_px=2.0):
        self.R, self.t = np.asarray(R, float), np.asarray(t, float)
        self.fovy, self.w, self.h, self.rms_px = float(fovy), int(w), int(h), float(rms_px)

    @property
    def f(self):
        return focal_px(self.fovy, self.h)

    @classmethod
    def load(cls, path=BASE_CLICK_CALIB):
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            c = json.load(fh)
        return cls(c["R_body_cam"], c["t_body_cam"], c["fovy"], *c["image_size"], c.get("rms_px", 2.0))

    def world_pose(self, kin):
        """Lens position and camera -> world rotation for the pose kin was last set to."""
        pb, Rb = body_pose(kin)
        return pb + Rb @ self.t, Rb @ self.R

    def ray(self, kin, u, v):
        """World ray (origin, unit direction) through pixel (u, v) (same pixel convention as click_goal_real.cam_ray)."""
        o, R = self.world_pose(kin)
        d = R @ np.array([(u - (self.w - 1) / 2) / self.f, (v - (self.h - 1) / 2) / self.f, 1.0])
        return o, d / np.linalg.norm(d)

    def project(self, kin, p):
        """Pixel (u, v) of world point p, or None if it is behind the camera."""
        o, R = self.world_pose(kin)
        pc = R.T @ (np.asarray(p) - o)
        if pc[2] < 1e-2:
            return None
        u, v = (self.w - 1) / 2 + self.f * pc[0] / pc[2], (self.h - 1) / 2 + self.f * pc[1] / pc[2]
        return int(np.clip(u, -1e4, 1e4)), int(np.clip(v, -1e4, 1e4))
