"""SO-101 pick & place in a replica of the real setup (../scene.xml): white block -> green note on a dark mat.

World layout (table, mat, note, block, lights) is read from ../scene.xml, so that file stays the one description
of the real setup. Everything is expressed in the ROBOT frame: origin = shoulder-pan axis on the table top,
+x = straight ahead into the mat, +z = up. The robot is mounted at the centre of the mat's front edge.

Cameras:
    top_camera   : the pose measured on the real camera by calibrate_top_camera.py (top_camera_calib.json).
                   Without that file: 0.19 m ahead, 0.50 m above the table, straight down, 45 deg vertical FOV.
    wrist_camera : on the fixed jaw, as in ../so101_ik.py (WRIST_CAM_POS / WRIST_CAM_TILT, 70 deg)

IK plans 8 joint keyframes (training labels); the arm itself is driven only in joint space by `move_to`.

    python real_env.py                    # watch random expert episodes (with randomisation)
    python real_env.py --headless -n 100  # expert success rate
"""
import os
import re
import sys
import json
import time
import argparse
from dataclasses import dataclass

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)  # appended: this folder's modules must win over same-named ones in the root
from so101_ik import (SO101_XML, IK, ARM_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_FORCE,  # noqa: E402
                      WRIST_CAM_POS, WRIST_CAM_TILT, quat_from_mat, quat_to_mat, top_down_rotation, grasp_yaw)

SCENE_XML = os.path.join(ROOT, "scene.xml")
HERE = os.path.dirname(os.path.abspath(__file__))
TOP_CAM_CALIB = os.path.join(HERE, "top_camera_calib.json")
PHASES = ["PRE_GRASP", "GRASP", "CLOSE", "LIFT", "PRE_PLACE", "PLACE", "RELEASE", "RETREAT"]


def _scene_dims():
    """Sizes and placements read from scene.xml, the one description of the real setup."""
    s = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(s)
    mujoco.mj_forward(s, d)
    gid = {n: s.geom(n).id for n in ("table_top", "dark_mat_geom", "green_note_geom", "white_block_geom")}
    table_top = d.geom_xpos[gid["table_top"]][2] + s.geom_size[gid["table_top"]][2]
    mat = gid["dark_mat_geom"]
    note_R = d.geom_xmat[gid["green_note_geom"]].reshape(3, 3)
    return {
        "table_top": table_top,
        "mat_center": d.geom_xpos[mat][:2].copy(),
        "mat_half": float(s.geom_size[mat][0]),
        "mat_top": float(d.geom_xpos[mat][2] + s.geom_size[mat][2] - table_top),
        "block_half": s.geom_size[gid["white_block_geom"]].copy(),
        "note_half": s.geom_size[gid["green_note_geom"]].copy(),
        "note_xy": d.geom_xpos[gid["green_note_geom"]][:2].copy(),
        "note_yaw": float(np.arctan2(note_R[1, 0], note_R[0, 0])),
    }


_DIMS = _scene_dims()
MAT_HALF = _DIMS["mat_half"]
MAT_TOP = _DIMS["mat_top"]             # mat top above the table, robot frame
BLOCK_HALF = _DIMS["block_half"]
NOTE_HALF = _DIMS["note_half"]
NOTE_TOP = MAT_TOP + 2 * NOTE_HALF[2]
# Robot mount in scene.xml coordinates: centre of the mat's front (-y) edge, on the table, facing +y
MOUNT_SCENE_POS = np.array([_DIMS["mat_center"][0], _DIMS["mat_center"][1] - MAT_HALF, _DIMS["table_top"]])

# TCP: between the jaws. Fixed jaw face at site z=0, so the block centre sits BLOCK_HALF + 3 mm clearance
# away from it (so101_ik.py uses 12.5 mm for its 25 mm cube).
TCP_OFFSET = np.array([-0.010, 0.0, BLOCK_HALF[0] + 0.003])

# TCP heights (robot frame). Grasp the upper part of the block; pads cover ~28 mm below the TCP + 10 mm.
GRASP_Z = MAT_TOP + 2 * BLOCK_HALF[2] - 0.020
HOVER_Z = MAT_TOP + 2 * BLOCK_HALF[2] + 0.050
LIFT_DZ = 0.060
PLACE_Z = GRASP_Z + (NOTE_TOP - MAT_TOP) + 0.003   # block bottom ~3 mm above the note when released
# Far away the 5-DOF wrist can't point straight down any more. Grasps must be vertical; releasing the
# block with a slight tilt is fine.
MAX_GRASP_TILT_DEG = 5.0
MAX_PLACE_TILT_DEG = 10.0
# The note's usual spot is near the edge of reach: when the note is far, aim slightly nearer the robot
# (smooth rule: pull = min(PLACE_PULL_MAX, max(0, r - PLACE_PULL_FROM)) towards the base).
PLACE_PULL_FROM = 0.27
PLACE_PULL_MAX = 0.015

# Where things may be placed (robot frame): the reachable band, inside the mat. Closer than 20 cm the
# arm's links come down next to the block and knock it over.
REACH = (0.20, 0.32)
MAT_MARGIN = 0.03
NOTE_DEFAULT_SCENE = _DIMS["note_xy"]           # scene.xml position
NOTE_DEFAULT_YAW = _DIMS["note_yaw"] - np.pi / 2  # in the robot frame (rotated 90 deg from scene.xml)
NOTE_DEFAULT_PROB = 0.6                         # usually on its spot, sometimes anywhere reachable
NOTE_DEFAULT_JITTER = 0.03
MIN_BLOCK_NOTE_DIST = 0.10

HOME_POS = np.array([0.18, 0.0, 0.16])
HOME_SEED = np.array([0.0, -1.0, 1.0, 1.2, 0.0])
PARK_PAN = np.radians(100)  # start pose: home with the base turned away, out of the top camera's view

# The real arm is black: matte black printed parts, dark grey servos (the SO-101 model ships yellow parts)
ARM_PART_RGBA = (0.07, 0.07, 0.075, 1.0)
ARM_SERVO_RGBA = (0.13, 0.13, 0.14, 1.0)
ARM_SPECULAR = 0.25                     # matte PLA, not glossy

TOP_CAM_POS = np.array([0.19, 0.0, 0.50])
TOP_CAM_FOVY = 45.0


def top_camera_pose():
    """Calibrated real top camera (calibrate_top_camera.py) if available, else the default straight-down one."""
    if os.path.exists(TOP_CAM_CALIB):
        with open(TOP_CAM_CALIB) as f:
            c = json.load(f)
        return {"pos": np.array(c["pos"]), "quat": np.array(c["quat"]), "fovy": float(c["fovy"])}
    down = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).astype(float)  # cam x=-y, y=+x, looks along -z
    return {"pos": TOP_CAM_POS, "quat": quat_from_mat(down), "fovy": TOP_CAM_FOVY}


def scene_to_robot_rot():
    """Rotation from scene.xml axes to robot axes (robot +x = scene +y, robot +y = scene -x)."""
    return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def scene_to_robot(p):
    return scene_to_robot_rot() @ (np.asarray(p, float) - MOUNT_SCENE_POS)


# ── Model construction ──────────────────────────────────────────────────────
def _add_robot_parts(spec):
    """Grasping pads, TCP site and wrist camera. Same as the robot part of so101_ik.build_model(),
    except TCP_OFFSET (block size)."""
    for body in spec.bodies:
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue  # visual geom
            if body.name in ("gripper", "moving_jaw_so101_v1"):
                g.contype, g.conaffinity = 0, 0   # convex hulls fill the jaw gap: grasp with pads instead
            else:
                g.contype, g.conaffinity = 2, 1
    spec.actuator("gripper").forcerange = [-GRIPPER_FORCE, GRIPPER_FORCE]

    site = spec.site("gripperframe")
    R_bs, p_bs = quat_to_mat(site.quat), np.array(site.pos)

    def site_to_body(p_site):
        return p_bs + R_bs @ np.asarray(p_site)

    gripper, jaw = spec.body("gripper"), spec.body("moving_jaw_so101_v1")
    gripper.add_site(name="tcp", pos=site_to_body(TCP_OFFSET), quat=site.quat,
                     size=[0.004, 0, 0], rgba=[1, 0, 0, 1], group=3)

    pad_kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, contype=2, conaffinity=1, condim=4,
                  friction=[1.5, 0.02, 0.002], solref=[0.01, 1], rgba=[0.9, 0.2, 0.2, 0.6], group=3, mass=0.001)
    gripper.add_geom(name="fixed_pad", pos=site_to_body([-0.021, 0, -0.003]), quat=site.quat,
                     size=[0.019, 0.008, 0.003], **pad_kw)
    tmp = spec.copy().compile()
    tmp_d = mujoco.MjData(tmp)
    mujoco.mj_forward(tmp, tmp_d)
    sid, jid = tmp.site("gripperframe").id, tmp.body("moving_jaw_so101_v1").id
    R_ws, p_ws = tmp_d.site_xmat[sid].reshape(3, 3), tmp_d.site_xpos[sid]
    R_wj, p_wj = tmp_d.xmat[jid].reshape(3, 3), tmp_d.xpos[jid]
    pad_center_w = p_ws + R_ws @ np.array([-0.021, 0, 0.019])
    jaw.add_geom(name="moving_pad", pos=R_wj.T @ (pad_center_w - p_wj),
                 quat=quat_from_mat(R_wj.T @ R_ws), size=[0.019, 0.008, 0.003], **pad_kw)

    tilt = np.deg2rad(WRIST_CAM_TILT)
    R_sc = np.column_stack([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).astype(float)
    Ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    gripper.add_camera(name="wrist_camera", pos=site_to_body(WRIST_CAM_POS),
                       quat=quat_from_mat(R_bs @ Ry.T @ R_sc), fovy=70)


def _add_scene_xml(spec):
    """Copy floor, table, mat, note, block, lights and headlight from scene.xml into the robot frame."""
    scene = mujoco.MjModel.from_xml_path(SCENE_XML)   # compiled: angles already in radians
    sd = mujoco.MjData(scene)
    mujoco.mj_forward(scene, sd)
    Rsr = scene_to_robot_rot()
    q_sr = quat_from_mat(Rsr)

    def to_robot_quat(xmat):
        return quat_from_mat(Rsr @ xmat.reshape(3, 3))

    wb = spec.worldbody
    for g in range(scene.ngeom):
        name = scene.geom(g).name
        body = scene.geom_bodyid[g]
        kw = dict(name=name, type=mujoco.mjtGeom(scene.geom_type[g]), size=scene.geom_size[g].copy(),
                  rgba=scene.geom_rgba[g].copy(), friction=scene.geom_friction[g].copy())
        if scene.body_jntnum[body] == 0:      # static: floor, table top, mat
            wb.add_geom(pos=scene_to_robot(sd.geom_xpos[g]), quat=to_robot_quat(sd.geom_xmat[g]), **kw)
        else:                                 # free bodies: green note, white block
            b = wb.add_body(name=scene.body(body).name, pos=scene_to_robot(sd.xpos[body]),
                            quat=to_robot_quat(sd.xmat[body]))
            b.add_freejoint(name=scene.joint(scene.body_jntadr[body]).name)
            b.add_geom(mass=scene.body_mass[body], condim=4, **kw)

    for i in range(scene.nlight):
        wb.add_light(name=scene.light(i).name, pos=scene_to_robot(scene.light_pos[i]),
                     dir=Rsr @ scene.light_dir[i], diffuse=scene.light_diffuse[i].copy(),
                     specular=scene.light_specular[i].copy(), castshadow=bool(scene.light_castshadow[i]))
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse, hl.specular = (scene.vis.headlight.ambient.copy(), scene.vis.headlight.diffuse.copy(),
                                           scene.vis.headlight.specular.copy())
    del q_sr


def build_real_model():
    spec = mujoco.MjSpec.from_file(SO101_XML)
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC   # grasp-friendly friction (as in so101_ik.py)
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480
    _add_robot_parts(spec)
    for mat in spec.materials:          # robot materials only: the scene geoms have plain rgba
        mat.rgba = ARM_SERVO_RGBA if mat.name.startswith("sts3215") else ARM_PART_RGBA
        mat.specular = ARM_SPECULAR
    _add_scene_xml(spec)

    spec.worldbody.add_camera(name="top_camera", **top_camera_pose())
    return spec.compile()


# ── Actions ─────────────────────────────────────────────────────────────────
@dataclass
class Keyframe:
    phase: str
    q: np.ndarray       # 5 arm joint angles (rad)
    gripper: float      # gripper command (rad)
    tcp_pos: np.ndarray


def gripper_word(g):
    return "open" if g > (GRIPPER_OPEN + GRIPPER_CLOSED) / 2 else "closed"


def format_action(q, gripper):
    return "j: " + " ".join(f"{v:.1f}" for v in np.degrees(q)) + f" | gripper: {gripper_word(gripper)}"


ACTION_RE = re.compile(r"j:\s*((?:-?\d+(?:\.\d+)?\s+){4}-?\d+(?:\.\d+)?)\s*\|\s*gripper:\s*(open|closed)")


def parse_action(text):
    m = ACTION_RE.search(text)
    if m is None:
        return None
    q = np.radians([float(v) for v in m.group(1).split()])
    return q, (GRIPPER_OPEN if m.group(2) == "open" else GRIPPER_CLOSED)


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ── Environment ─────────────────────────────────────────────────────────────
class RealSetupEnv:
    def __init__(self, randomize=True):
        m = self.model = build_real_model()
        self.data = mujoco.MjData(m)
        self.ik = IK(m)
        self.randomize = randomize

        act = {m.actuator(i).name: i for i in range(m.nu)}
        self.arm_act = np.array([act[j] for j in ARM_JOINTS])
        self.grip_act = act["gripper"]
        self.arm_qadr = self.ik.qadr
        self.arm_lo, self.arm_hi = self.ik.lo, self.ik.hi
        self.grip_qadr = m.jnt_qposadr[m.joint("gripper").id]
        self.block_body = m.body("white_block").id
        self.note_body = m.body("green_note").id
        self.block_qadr = m.jnt_qposadr[m.joint("white_block_joint").id]
        self.note_qadr = m.jnt_qposadr[m.joint("green_note_joint").id]
        self.top_cam = m.camera("top_camera").id

        self.q_seed, err, _ = self.ik.solve(HOME_POS, top_down_rotation(0.0), HOME_SEED, iters=300)
        assert err < 2e-3, f"home pose unreachable ({err * 1000:.1f} mm)"
        self.q_home = self.q_seed.copy()
        self.q_home[0] = PARK_PAN

        # Nominal look, restored before each randomisation
        self._nominal = {
            "geom_rgba": m.geom_rgba.copy(), "light_diffuse": m.light_diffuse.copy(),
            "light_pos": m.light_pos.copy(), "cam_pos": m.cam_pos.copy(), "cam_quat": m.cam_quat.copy(),
            "cam_fovy": m.cam_fovy.copy(), "headlight_ambient": m.vis.headlight.ambient.copy(),
            "mat_rgba": m.mat_rgba.copy(),
        }

    # ── Task setup ─────────────────────────────────────────────────────────
    @staticmethod
    def _in_mat(xy):
        s = scene_to_robot_rot()[:2, :2].T @ xy + MOUNT_SCENE_POS[:2]   # robot xy -> scene xy
        return np.all(np.abs(s) <= MAT_HALF - MAT_MARGIN)

    def _sample_reachable(self, rng):
        while True:
            r, a = rng.uniform(*REACH), rng.uniform(-np.pi / 2, np.pi / 2)
            xy = r * np.array([np.cos(a), np.sin(a)])
            if self._in_mat(xy):
                return xy

    def sample_task(self, rng):
        if rng.random() < NOTE_DEFAULT_PROB:
            note_xy = scene_to_robot([*NOTE_DEFAULT_SCENE, MOUNT_SCENE_POS[2]])[:2] + \
                rng.uniform(-NOTE_DEFAULT_JITTER, NOTE_DEFAULT_JITTER, 2)
            note_yaw = NOTE_DEFAULT_YAW + rng.uniform(-0.2, 0.2)
        else:
            note_xy, note_yaw = self._sample_reachable(rng), rng.uniform(-np.pi, np.pi)
        while True:
            block_xy = self._sample_reachable(rng)
            if np.linalg.norm(block_xy - note_xy) >= MIN_BLOCK_NOTE_DIST:
                break
        return {"block_xy": block_xy, "block_yaw": rng.uniform(-np.pi, np.pi),
                "note_xy": note_xy, "note_yaw": note_yaw}

    def _apply_randomization(self, rng):
        m, nom = self.model, self._nominal
        m.geom_rgba[:] = nom["geom_rgba"]
        m.light_diffuse[:] = nom["light_diffuse"]
        m.light_pos[:] = nom["light_pos"]
        m.cam_pos[:], m.cam_quat[:], m.cam_fovy[:] = nom["cam_pos"], nom["cam_quat"], nom["cam_fovy"]
        m.vis.headlight.ambient[:] = nom["headlight_ambient"]
        m.mat_rgba[:] = nom["mat_rgba"]
        if not self.randomize:
            return
        # Black plastic looks anywhere from near-black to dark grey depending on the light
        m.mat_rgba[:, :3] = np.clip(nom["mat_rgba"][:, :3] + rng.uniform(-0.03, 0.06), 0, 1)
        # (brightness scale range, per-channel tint). Under real light / auto-exposure the "dark" mat
        # can look anywhere from dark to light grey, so its brightness varies a lot.
        for name, scale, jitter in (("table_top", (0.85, 1.05), 0.04), ("dark_mat_geom", (0.8, 2.4), 0.04),
                                    ("green_note_geom", (0.85, 1.15), 0.05), ("white_block_geom", (0.85, 1.05), 0.04)):
            g = m.geom(name).id
            m.geom_rgba[g, :3] = np.clip(nom["geom_rgba"][g, :3] * rng.uniform(*scale)
                                         + rng.uniform(-jitter, jitter, 3), 0, 1)
        m.light_diffuse[:] = nom["light_diffuse"] * rng.uniform(0.7, 1.3)
        m.light_pos[:] = nom["light_pos"] + rng.uniform(-0.15, 0.15, nom["light_pos"].shape) * [1, 1, 0]
        m.vis.headlight.ambient[:] = nom["headlight_ambient"] * rng.uniform(0.7, 1.5)
        # The real top camera won't be mounted perfectly: ~1 cm / ~1.5 deg / ~1 deg FOV
        c = self.top_cam
        m.cam_pos[c] = nom["cam_pos"][c] + rng.uniform(-0.01, 0.01, 3)
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, rng.normal(size=3) / np.linalg.norm(rng.normal(size=3) + 1e-9),
                                  np.radians(rng.uniform(0, 1.5)))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, nom["cam_quat"][c], dq)
        m.cam_quat[c] = q
        m.cam_fovy[c] = nom["cam_fovy"][c] + rng.uniform(-1, 1)

    def reset(self, task, rng=None):
        m, d = self.model, self.data
        self._apply_randomization(rng if rng is not None else np.random.default_rng())
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qadr] = self.q_home
        d.ctrl[self.arm_act] = self.q_home
        d.qpos[self.grip_qadr] = GRIPPER_OPEN
        d.ctrl[self.grip_act] = GRIPPER_OPEN
        for qadr, xy, z, yaw in ((self.block_qadr, task["block_xy"], MAT_TOP + BLOCK_HALF[2], task["block_yaw"]),
                                 (self.note_qadr, task["note_xy"], MAT_TOP + NOTE_HALF[2], task["note_yaw"])):
            d.qpos[qadr:qadr + 3] = [*xy, z]
            d.qpos[qadr + 3:qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        self.step(int(0.3 / m.opt.timestep))  # settle

    def block_pose(self):
        return self.data.xpos[self.block_body].copy(), yaw_of(self.data.xquat[self.block_body])

    def note_pose(self):
        return self.data.xpos[self.note_body].copy(), yaw_of(self.data.xquat[self.note_body])

    def arm_q(self):
        return self.data.qpos[self.arm_qadr].copy()

    def forward_tcp(self, q):
        return self.ik.fk(q)[0]

    # ── Planning ───────────────────────────────────────────────────────────
    def place_target(self):
        """Note centre, pulled slightly towards the base when the note is far (see PLACE_PULL_*)."""
        note_pos, _ = self.note_pose()
        r = np.linalg.norm(note_pos[:2])
        target = note_pos.copy()
        target[:2] -= min(PLACE_PULL_MAX, max(0.0, r - PLACE_PULL_FROM)) * note_pos[:2] / r
        return target

    def plan_keyframes(self):
        """8 joint keyframes from the true block / note pose. None if any pose is out of reach."""
        block_pos, block_yaw = self.block_pose()
        note_pos = self.place_target()
        gy = grasp_yaw(block_pos, block_yaw)
        R_pick = top_down_rotation(gy)
        R_place = top_down_rotation(grasp_yaw(note_pos, gy))
        bx, by = block_pos[:2]
        nx, ny = note_pos[:2]
        targets = [
            ("PRE_GRASP", (bx, by, HOVER_Z),           R_pick,  GRIPPER_OPEN),
            ("GRASP",     (bx, by, GRASP_Z),           R_pick,  GRIPPER_OPEN),
            ("CLOSE",     (bx, by, GRASP_Z),           R_pick,  GRIPPER_CLOSED),
            ("LIFT",      (bx, by, GRASP_Z + LIFT_DZ), R_pick,  GRIPPER_CLOSED),
            ("PRE_PLACE", (nx, ny, GRASP_Z + LIFT_DZ), R_place, GRIPPER_CLOSED),
            ("PLACE",     (nx, ny, PLACE_Z),           R_place, GRIPPER_CLOSED),
            ("RELEASE",   (nx, ny, PLACE_Z),           R_place, GRIPPER_OPEN),
            ("RETREAT",   (nx, ny, HOVER_Z),           R_place, GRIPPER_OPEN),
        ]
        keyframes, q_seed = [], self.q_seed.copy()
        for phase, pos, R, grip in targets:
            pos = np.array(pos)
            q_seed, err_pos, err_rot = self.ik.solve(pos, R, q_seed, iters=300)
            max_tilt = {"GRASP": MAX_GRASP_TILT_DEG, "CLOSE": MAX_GRASP_TILT_DEG,
                        "PLACE": MAX_PLACE_TILT_DEG, "RELEASE": MAX_PLACE_TILT_DEG}.get(phase, 180.0)
            if err_pos > 2e-3 or np.degrees(err_rot) > max_tilt:
                return None
            keyframes.append(Keyframe(phase, q_seed.copy(), grip, pos))
        return keyframes

    # ── Joint-space execution ──────────────────────────────────────────────
    def step(self, n, viewer=None):
        dt = self.model.opt.timestep
        for i in range(n):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None and i % 10 == 0:
                viewer.sync()
                time.sleep(10 * dt)

    def move_to(self, q_target, gripper, max_speed=1.0, viewer=None):
        d, dt = self.data, self.model.opt.timestep
        q_start = d.ctrl[self.arm_act].copy()
        q_target = np.clip(q_target, self.arm_lo, self.arm_hi)
        gripper_changed = abs(d.ctrl[self.grip_act] - gripper) > 1e-6
        d.ctrl[self.grip_act] = gripper
        n = int(max(np.max(np.abs(q_target - q_start)) / max_speed, 0.4) / dt)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)  # smoothstep
            d.ctrl[self.arm_act] = q_start + a * (q_target - q_start)
            self.step(1, viewer)
        self.step(int((0.8 if gripper_changed else 0.2) / dt), viewer)

    def is_success(self):
        """Block upright, resting, with its centre over the note."""
        block_pos, _ = self.block_pose()
        note_pos, note_yaw = self.note_pose()
        local = rot_z(-note_yaw) @ (block_pos - note_pos)
        on_note = abs(local[0]) < NOTE_HALF[0] and abs(local[1]) < NOTE_HALF[1]
        upright = self.data.xmat[self.block_body].reshape(3, 3)[2, 2] > np.cos(np.radians(15))
        resting = block_pos[2] < NOTE_TOP + BLOCK_HALF[2] + 0.006
        return bool(on_note and upright and resting)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-randomize", action="store_true")
    args = ap.parse_args()

    env = RealSetupEnv(randomize=not args.no_randomize)
    rng = np.random.default_rng(args.seed)
    stats = {"success": 0, "fail": 0, "ik_fail": 0}

    def run(viewer=None):
        for ep in range(args.episodes):
            env.reset(env.sample_task(rng), rng)
            keyframes = env.plan_keyframes()
            if keyframes is None:
                stats["ik_fail"] += 1
                print(f"ep {ep}: out of reach, skipped")
                continue
            for kf in keyframes:
                env.move_to(kf.q, kf.gripper, viewer=viewer)
                if viewer is not None and not viewer.is_running():
                    return
            ok = env.is_success()
            stats["success" if ok else "fail"] += 1
            print(f"ep {ep}: {'success' if ok else 'FAIL'}")

    if args.headless:
        run()
    else:
        from mujoco import viewer
        with viewer.launch_passive(env.model, env.data) as v:
            v.cam.lookat[:] = [0.25, 0.0, 0.05]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9, 200, -30
            run(v)
    print(f"success {stats['success']}/{stats['success'] + stats['fail']} (out of reach: {stats['ik_fail']})")


if __name__ == "__main__":
    main()
