"""
SO-101 arm — inverse kinematics pick & place in MuJoCo.

Same idea as ik_implemnetion.py (Panda), adapted to the 5-DOF SO-101:
  * IK is a 6-D (position + orientation) damped-least-squares solve over the
    5 arm joints. A 5-DOF arm cannot reach an arbitrary 6-D pose, but a
    *top-down* pose is special: when the gripper points straight down,
    wrist_roll rotates about the vertical axis, so position + "point down" +
    yaw are all reachable.
  * IK runs on its own MjData (pure kinematics), so the joint targets never
    depend on how well the physics is tracking.
  * The cartesian target moves along a min-jerk path between waypoints, and
    the IK is warm-started every control step -> straight, smooth motions.

Usage:
  python so101_ik.py                 # viewer + camera windows
  python so101_ik.py --no-cams       # viewer only
  python so101_ik.py --headless      # no GUI, prints result (for testing)
  python so101_ik.py --seed 3        # random cube position / yaw
"""

import argparse
import os
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO101_XML = os.path.join(
    HERE, "so_101_arm/simulation_URDF/Assets/Simulation/SO101/so101_new_calib.xml"
)

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_OPEN = 0.8      # rad, moving-jaw joint (~0.17 = closed, 1.74 = max open)
GRIPPER_CLOSED = -0.17  # commanding past contact squeezes the cube
GRIPPER_FORCE = 1.0     # N·m, gripper actuator force limit

CUBE_HALF = 0.0125      # 25 mm cube
# TCP = point between the jaws, expressed in the gripperframe site frame
# (site x = approach direction, site z = jaw closing direction, fixed jaw at z=0)
TCP_OFFSET = np.array([-0.010, 0.0, CUBE_HALF])


WRIST_CAM_POS = [-0.06, 0.0, -0.055]  # in gripperframe site coordinates
WRIST_CAM_TILT = 25                    # deg

# ── Scene construction ───────────────────────────────────────────────────────
def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def build_model(cube_pos, cube_yaw, place_pos):
    spec = mujoco.MjSpec.from_file(SO101_XML)
    spec.option.timestep = 0.002
    # Elliptic friction cones + high impratio -> objects don't slip out of
    # the gripper (MuJoCo's recommended setting for grasping).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 480

    # Robot mesh collisions: contype=2 / conaffinity=1 -> collide with the
    # world (floor, cube) but never with each other. The gripper / jaw meshes
    # are turned off completely: their convex hulls fill the gap between the
    # fingers, so we grasp with box pads instead.
    for body in spec.bodies:
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue  # visual geom
            if body.name in ("gripper", "moving_jaw_so101_v1"):
                g.contype, g.conaffinity = 0, 0
            else:
                g.contype, g.conaffinity = 2, 1

    # The STS3215 position servo (kp≈1000, 3.35 N·m) squeezes a light cube so
    # hard that MuJoCo's soft contacts let the pads sink through it. A gentler
    # grip holds it much better.
    spec.actuator("gripper").forcerange = [-GRIPPER_FORCE, GRIPPER_FORCE]

    site = spec.site("gripperframe")
    R_bs = quat_to_mat(site.quat)          # gripper-body <- site rotation
    p_bs = np.array(site.pos)

    def site_to_body(p_site):
        return p_bs + R_bs @ np.asarray(p_site)

    gripper = spec.body("gripper")
    jaw = spec.body("moving_jaw_so101_v1")

    # TCP site between the jaws (this is what the IK drives)
    gripper.add_site(name="tcp", pos=site_to_body(TCP_OFFSET), quat=site.quat,
                     size=[0.004, 0, 0], rgba=[1, 0, 0, 1], group=3)

    # Finger pads. Sizes are in the site frame (x along fingers, z = closing).
    pad_kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, contype=2, conaffinity=1,
                  condim=4, friction=[1.5, 0.02, 0.002], solref=[0.01, 1],
                  rgba=[0.9, 0.2, 0.2, 0.6], group=3, mass=0.001)
    gripper.add_geom(name="fixed_pad", pos=site_to_body([-0.021, 0, -0.003]),
                     quat=site.quat, size=[0.019, 0.008, 0.003], **pad_kw)

    # The moving pad lives in the jaw body: map the site-frame box (taken at
    # gripper = 0) into the jaw frame.
    tmp = spec.copy().compile()
    tmp_d = mujoco.MjData(tmp)
    mujoco.mj_forward(tmp, tmp_d)
    sid = tmp.site("gripperframe").id
    jid = tmp.body("moving_jaw_so101_v1").id
    R_ws, p_ws = tmp_d.site_xmat[sid].reshape(3, 3), tmp_d.site_xpos[sid]
    R_wj, p_wj = tmp_d.xmat[jid].reshape(3, 3), tmp_d.xpos[jid]
    pad_center_w = p_ws + R_ws @ np.array([-0.021, 0, 0.019])
    jaw.add_geom(name="moving_pad", pos=R_wj.T @ (pad_center_w - p_wj),
                 quat=quat_from_mat(R_wj.T @ R_ws), size=[0.019, 0.008, 0.003], **pad_kw)

    # Wrist camera: outside the fixed jaw, looking along the approach axis.
    # Camera looks down its -z; axes in site frame: x=+y_s, y=-z_s, z=-x_s.
    tilt = np.deg2rad(WRIST_CAM_TILT)  # tilt towards the jaw centre
    R_sc = np.column_stack([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).astype(float)
    Ry = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    gripper.add_camera(name="wrist_camera", pos=site_to_body(WRIST_CAM_POS),
                       quat=quat_from_mat(R_bs @ Ry.T @ R_sc), fovy=70)

    # ── World ──
    wb = spec.worldbody
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4],
                     rgb2=[0.1, 0.2, 0.3], width=300, height=300,
                     mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8])
    spec.add_material(name="grid", texrepeat=[8, 8], texuniform=True, reflectance=0.1,
                      textures=["", "grid"])
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.3, 0.5, 0.7],
                     rgb2=[0, 0, 0], width=512, height=3072)
    wb.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(pos=[0.6, -0.6, 0.8], dir=[-0.6, 0.6, -0.8], diffuse=[0.4, 0.4, 0.4])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                material="grid", friction=[1.0, 0.01, 0.001])

    cube = wb.add_body(name="cube", pos=cube_pos,
                       quat=[np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  mass=0.03, condim=4, friction=[1.5, 0.02, 0.002], rgba=[0.85, 0.15, 0.15, 1])

    wb.add_geom(name="place_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[place_pos[0], place_pos[1], 0.0005], size=[0.025, 0.0005, 0],
                rgba=[0.1, 0.8, 0.2, 0.6], contype=0, conaffinity=0)

    target = wb.add_body(name="ik_target", mocap=True)
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0, 0],
                    rgba=[0.1, 0.4, 1, 0.5], contype=0, conaffinity=0)

    def look_at(pos, at, up=(0, 0, 1)):
        z = np.asarray(pos, float) - np.asarray(at, float)
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        return quat_from_mat(np.column_stack([x, np.cross(z, x), z]))

    wb.add_camera(name="top_camera", pos=[0.2, 0, 0.75],
                  quat=look_at([0.2, 0, 0.75], [0.2, 0, 0], up=(1, 0, 0)), fovy=50)
    wb.add_camera(name="front_camera", pos=[0.65, 0, 0.35],
                  quat=look_at([0.65, 0, 0.35], [0.15, 0, 0.05]), fovy=55)

    return spec.compile()


# ── IK ───────────────────────────────────────────────────────────────────────
def top_down_rotation(yaw):
    """Gripper frame (site axes) pointing straight down; jaws close along yaw."""
    x = np.array([0.0, 0.0, -1.0])                    # approach: down
    z = np.array([np.cos(yaw), np.sin(yaw), 0.0])     # jaw closing direction
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def grasp_yaw(cube_pos, cube_yaw):
    """Pick the cube face direction closest to the arm's natural wrist roll
    (jaws closing along the radial direction) so wrist_roll stays small."""
    radial = np.arctan2(cube_pos[1], cube_pos[0])
    cands = cube_yaw + np.arange(-4, 5) * np.pi / 2
    return cands[np.argmin(np.abs(np.angle(np.exp(1j * (cands - radial)))))]


class IK:
    def __init__(self, model, site="tcp", joints=ARM_JOINTS):
        self.m = model
        self.d = mujoco.MjData(model)          # private copy: kinematics only
        self.site = model.site(site).id
        jids = [model.joint(j).id for j in joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0]
        self.hi = model.jnt_range[jids, 1]
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    @staticmethod
    def _dls_pinv(J, damping):
        """Damped pseudo-inverse  Jᵀ (J Jᵀ + λI)⁻¹."""
        return J.T @ np.linalg.inv(J @ J.T + damping * np.eye(J.shape[0]))

    def solve(self, target_pos, target_R, q_init, iters=100, damping=1e-4,
              max_step=0.25, tol=1e-5):
        """Task-priority damped least squares (5 joints).

        Priority 1 — TCP position (3 eqns):    dq₁ = J_p⁺ e_p
        Priority 2 — orientation in the null space of the position task:
            N  = I − J_p⁺ J_p
            dq = dq₁ + N (J_r N)⁺ (e_r − J_r dq₁)

        Where the full pose is reachable (top-down near the table) both are met
        exactly; higher up the gripper tilts as little as the wrist limits allow.
        """
        q = np.clip(np.array(q_init, dtype=float), self.lo, self.hi)
        for _ in range(iters):
            pos, R = self.fk(q)
            mujoco.mj_comPos(self.m, self.d)   # needed by mj_jacSite
            e_p = target_pos - pos
            # world-frame orientation error (small-angle axis * angle)
            e_r = 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            Jp, Jr = self.jacp[:, self.dadr], self.jacr[:, self.dadr]

            # Joint limits: a joint sitting on a limit and being pushed further
            # out is removed from both tasks, so the others take over its work.
            free = np.ones(len(q))
            for _pass in range(len(q)):
                F = np.diag(free)
                Jp_pinv = self._dls_pinv(Jp @ F, damping)
                dq = Jp_pinv @ e_p
                N = F - Jp_pinv @ (Jp @ F)
                dq = dq + N @ self._dls_pinv(Jr @ N, 1e-2) @ (e_r - Jr @ dq)
                blocked = ((q <= self.lo + 1e-6) & (dq < 0)) | ((q >= self.hi - 1e-6) & (dq > 0))
                if not np.any(blocked & (free > 0)):
                    break
                free[blocked] = 0.0

            n = np.linalg.norm(dq)
            if n < tol:
                break
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        pos, R = self.fk(q)
        err_pos = np.linalg.norm(target_pos - pos)
        err_rot = np.linalg.norm(0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3)))
        return q, err_pos, err_rot


# ── Trajectory helpers ───────────────────────────────────────────────────────
def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def slerp_R(R0, R1, s):
    q0, q1 = quat_from_mat(R0), quat_from_mat(R1)
    dv = np.zeros(3)
    mujoco.mju_subQuat(dv, q1, q0)            # q0 * exp(dv) = q1
    q = q0.copy()
    mujoco.mju_quatIntegrate(q, dv, s)
    return quat_to_mat(q)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-cams", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # ── STEP 1: Scene (cube + place location) ───────────────────────────────
    if args.seed is None:
        cube_xy, cube_yaw, place_xy = np.array([0.22, -0.08]), 0.3, np.array([0.20, 0.12])
    else:
        rng = np.random.default_rng(args.seed)
        r, a = rng.uniform(0.16, 0.26), rng.uniform(-0.7, -0.15)
        cube_xy, cube_yaw = r * np.array([np.cos(a), np.sin(a)]), rng.uniform(-np.pi, np.pi)
        r, a = rng.uniform(0.16, 0.26), rng.uniform(0.15, 0.7)
        place_xy = r * np.array([np.cos(a), np.sin(a)])
    cube_pos = np.array([cube_xy[0], cube_xy[1], CUBE_HALF])
    place_pos = np.array([place_xy[0], place_xy[1], CUBE_HALF + 0.002])

    model = build_model(cube_pos, cube_yaw, place_pos)
    data = mujoco.MjData(model)
    ik = IK(model)
    act = {model.actuator(i).name: i for i in range(model.nu)}
    arm_act = [act[j] for j in ARM_JOINTS]
    grip_act = act["gripper"]
    cube_id = model.body("cube").id
    mocap_id = model.body("ik_target").mocapid[0]

    # ── STEP 2: Home pose (arm folded up, gripper pointing down) ────────────
    home_pos = np.array([0.18, 0.0, 0.16])
    home_R = top_down_rotation(0.0)
    q_home, ep, er = ik.solve(home_pos, home_R, np.array([0, -1.0, 1.0, 1.2, 0]), iters=300)
    print(f"Home IK: q={np.round(q_home, 3)}  pos_err={ep*1000:.2f}mm")
    data.qpos[ik.qadr] = q_home
    data.ctrl[arm_act] = q_home
    data.ctrl[grip_act] = GRIPPER_OPEN
    data.qpos[model.joint("gripper").qposadr[0]] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    # ── STEP 3: Waypoints (TCP pose + gripper command) ──────────────────────
    gy = grasp_yaw(cube_pos, cube_yaw)
    place_yaw = grasp_yaw(place_pos, gy)
    R_pick, R_place = top_down_rotation(gy), top_down_rotation(place_yaw)
    up = np.array([0, 0, 1.0])
    waypoints = [  # (name, position, rotation, gripper, wait-after [s])
        ("pre-grasp",  cube_pos + 0.05 * up,           R_pick,  GRIPPER_OPEN,   0.0),
        ("grasp",      cube_pos + 0.002 * up,          R_pick,  GRIPPER_OPEN,   0.2),
        ("close",      cube_pos + 0.002 * up,          R_pick,  GRIPPER_CLOSED, 0.8),
        ("lift",       cube_pos + 0.08 * up,           R_pick,  GRIPPER_CLOSED, 0.0),
        ("above drop", place_pos + 0.08 * up,          R_place, GRIPPER_CLOSED, 0.0),
        ("lower",      place_pos + 0.005 * up,         R_place, GRIPPER_CLOSED, 0.2),
        ("release",    place_pos + 0.005 * up,         R_place, GRIPPER_OPEN,   0.5),
        ("retreat",    place_pos + 0.06 * up,          R_place, GRIPPER_OPEN,   0.0),
        ("home",       home_pos,                       home_R,  GRIPPER_OPEN,   0.0),
    ]

    # Check every waypoint is reachable before moving
    q_seed = q_home.copy()
    for name, p, R, _, _ in waypoints:
        q_seed, ep, er = ik.solve(p, R, q_seed, iters=300)
        flag = "OK" if ep < 2e-3 else "UNREACHABLE"
        tilt = "" if er < 1e-2 else f"  (tilted {np.degrees(er):.0f}° from target orientation)"
        print(f"  {name:11s} q={np.round(q_seed, 2)}  pos_err={ep*1000:5.2f}mm  {flag}{tilt}")

    # ── STEP 4: Control loop parameters ─────────────────────────────────────
    ctrl_dt = 0.01
    n_sub = int(round(ctrl_dt / model.opt.timestep))
    speed = 0.15                     # m/s along the cartesian path
    q_des = q_home.copy()

    state = 0
    seg_start_pos, seg_start_R = home_pos.copy(), home_R.copy()
    seg_t, wait_t = 0.0, None

    def seg_duration(i):
        p0 = seg_start_pos
        return max(0.6, np.linalg.norm(waypoints[i][1] - p0) / speed)

    seg_T = seg_duration(0)
    print(f"→ {waypoints[0][0]}")

    renderers = []
    if not args.headless and not args.no_cams:
        import cv2
        renderers = [(c, mujoco.Renderer(model, 240, 320))
                     for c in ("wrist_camera", "top_camera", "front_camera")]

    def step():
        """One 10 ms control step. Returns False when the sequence is finished."""
        nonlocal state, seg_start_pos, seg_start_R, seg_t, seg_T, wait_t, q_des
        if state >= len(waypoints):
            mujoco.mj_step(model, data, n_sub)
            return False

        name, p_goal, R_goal, grip, wait = waypoints[state]

        # Cartesian target on a min-jerk path
        seg_t += ctrl_dt
        s = min_jerk(seg_t / seg_T)
        p_tgt = seg_start_pos + s * (p_goal - seg_start_pos)
        R_tgt = slerp_R(seg_start_R, R_goal, s)
        data.mocap_pos[mocap_id] = p_tgt

        # IK warm-started from last solution
        q_des, _, _ = ik.solve(p_tgt, R_tgt, q_des, iters=15)
        data.ctrl[arm_act] = q_des
        data.ctrl[grip_act] = grip

        mujoco.mj_step(model, data, n_sub)

        # Segment done when the path is finished and the robot has caught up
        tcp = data.site_xpos[ik.site]
        if seg_t >= seg_T and np.linalg.norm(p_goal - tcp) < 0.01:
            if wait_t is None:
                wait_t = 0.0
            wait_t += ctrl_dt
            if wait_t >= wait:
                print(f"✓ {name} reached (tcp err {np.linalg.norm(p_goal - tcp)*1000:.1f} mm)")
                state += 1
                seg_start_pos, seg_start_R = p_goal.copy(), R_goal.copy()
                seg_t, wait_t = 0.0, None
                if state < len(waypoints):
                    seg_T = seg_duration(state)
                    print(f"→ {waypoints[state][0]}")
        elif seg_t > seg_T + 3.0:
            print(f"✗ timeout on '{name}' (tcp err {np.linalg.norm(p_goal - tcp)*1000:.1f} mm), skipping")
            state += 1
            seg_start_pos, seg_start_R = p_goal.copy(), R_goal.copy()
            seg_t, wait_t = 0.0, None
            if state < len(waypoints):
                seg_T = seg_duration(state)
        return True

    def report():
        cp = data.xpos[cube_id]
        d_xy = np.linalg.norm(cp[:2] - place_pos[:2])
        ok = d_xy < 0.02 and cp[2] < 0.03
        print(f"Cube final pos {np.round(cp, 3)}, {d_xy*1000:.1f} mm from target → "
              f"{'SUCCESS' if ok else 'FAILED'}")

    if args.headless:
        for _ in range(6000):
            if not step():
                break
        mujoco.mj_step(model, data, 50)
        report()
        return

    from mujoco import viewer
    done_reported = False
    with viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = [0.18, 0.0, 0.05]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.7, 150, -25
        while v.is_running():
            t0 = time.time()
            running = step()
            if not running and not done_reported:
                report()
                done_reported = True
            v.sync()

            for cam, r in renderers:
                r.update_scene(data, camera=cam)
                cv2.imshow(cam, cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
            if renderers:
                cv2.waitKey(1)

            time.sleep(max(0.0, ctrl_dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
