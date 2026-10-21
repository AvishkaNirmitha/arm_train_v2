import mujoco
from mujoco import viewer
import numpy as np
import time
import cv2

model = mujoco.MjModel.from_xml_path("mujoco_pick_place/franka_emika_panda/mjx_single_cube.xml")
data = mujoco.MjData(model)

# ── STEP 1: Find Joint IDs ──────────────────────────────────────────────────
joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
dof_ids = [
    model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    for name in joint_names
]
print("DOF IDs:", dof_ids)

# ── STEP 2: Find End-Effector ───────────────────────────────────────────────
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
print("End-effector ID:", ee_id)

# Set a nice start pose
start_angles = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
for i, name in enumerate(joint_names):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = start_angles[i]

mujoco.mj_forward(model, data)
print("EE start position:", data.site_xpos[ee_id])

# ── STEP 3: Define Target and State Machine ─────────────────────────────────
state = 0

# Dynamically get the box position so we can pick it up anywhere
box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
box_pos = data.xpos[box_id].copy()

# Define where to drop it (e.g., 30cm to the side of where it started)
drop_pos = box_pos.copy()
drop_pos[1] += 0.3  # Add 0.3 to the Y-axis

waypoints = [
    np.array([box_pos[0], box_pos[1], 0.15]),  # 0: Pre-grasp (above the box)
    np.array([box_pos[0], box_pos[1], 0.045]), # 1: Grasp (lower around the box)
    np.array([box_pos[0], box_pos[1], 0.045]), # 2: Close gripper (wait)
    np.array([box_pos[0], box_pos[1], 0.25]),  # 3: Lift (up with the box)
    np.array([drop_pos[0], drop_pos[1], 0.25]),  # 4: Move above drop location
    np.array([drop_pos[0], drop_pos[1], 0.05]),  # 5: Lower to drop surface
    np.array([drop_pos[0], drop_pos[1], 0.05]),  # 6: Open gripper (wait)
    np.array([drop_pos[0], drop_pos[1], 0.25]),  # 7: Move back up
]
gripper_states = [0.04, 0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04]  # 0.04 is open, 0.0 is closed
target_pos = waypoints[0]
wait_time = 0.0

# ── IK Parameters ──────────────────────────────────────────────────────────
damping    = 1e-3
dt         = 0.01
q_desired  = data.qpos.copy()

# ── Initialize Depth Camera Renderer ───────────────────────────────────────
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.enable_depth_rendering()

# ── Initialize RGB Camera Renderer ─────────────────────────────────────────
renderer_rgb = mujoco.Renderer(model, height=480, width=640)

# ── Initialize Top Camera Renderer ─────────────────────────────────────────
renderer_top = mujoco.Renderer(model, height=480, width=640)

# ── Initialize Front Camera Renderer ───────────────────────────────────────
renderer_front = mujoco.Renderer(model, height=480, width=640)

# ── STEP 4–6: Main loop ─────────────────────────────────────────────────────
with viewer.launch_passive(model, data) as v:
    while v.is_running():

        # ── STEP 3: Compute Error ───────────────────────────────────────────
        current_pos = data.site_xpos[ee_id].copy()
        error = target_pos - current_pos
        dist  = np.linalg.norm(error)

        if dist < 0.01:
            if state == 0:
                print("✓ Pre-grasp reached! Lowering...")
                state = 1
                target_pos = waypoints[state]
            elif state == 1:
                print("✓ Grasp position reached! Closing gripper...")
                state = 2
                target_pos = waypoints[state]
                wait_time = time.time()
            elif state == 2:
                if time.time() - wait_time > 0.5: # Wait 0.5s for gripper to close
                    print("✓ Gripper closed! Lifting...")
                    state = 3
                    target_pos = waypoints[state]
            elif state == 3:
                print("✓ Lifted! Moving to drop location...")
                state = 4
                target_pos = waypoints[state]
            elif state == 4:
                print("✓ Above drop location! Lowering...")
                state = 5
                target_pos = waypoints[state]
            elif state == 5:
                print("✓ Drop position reached! Opening gripper...")
                state = 6
                target_pos = waypoints[state]
                wait_time = time.time()
            elif state == 6:
                if time.time() - wait_time > 0.5: # Wait 0.5s for gripper to open
                    print("✓ Gripper opened! Moving back up...")
                    state = 7
                    target_pos = waypoints[state]
            elif state == 7:
                # Target reached, stay here
                pass
        
        # Apply gripper control
        if state < len(gripper_states):
            data.ctrl[7] = gripper_states[state]

        # ── STEP 4: Build Jacobian ──────────────────────────────────────────
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, ee_id)
        #                               ^^^  ^^^^
        #                           pos jac  rot jac (None = position only)

        # ── STEP 5: Solve IK (Damped Least Squares) ─────────────────────────
        #   dq = Jᵀ (J Jᵀ + λI)⁻¹ · error
        jjt  = jac @ jac.T                          # 3×3
        diag = damping * np.eye(3)                  # 3×3 regularisation
        dq   = jac.T @ np.linalg.solve(jjt + diag, error)  # nv,

        # ── STEP 6: Integrate & Apply ────────────────────────────────────────
        mujoco.mj_integratePos(model, q_desired, dq, dt)
        data.ctrl[:7] = q_desired[dof_ids]

        # Sub-step physics for stability
        model.opt.timestep = 0.002
        for _ in range(5):                          # 5 × 2ms = 10ms control step
            mujoco.mj_step(model, data)

        v.sync()

        # Render and display depth camera
        renderer.update_scene(data, camera="wrist_camera")
        depth_image = renderer.render()
        
        # Depth is in meters. Normalize to 0-255 for visualization (clip max 2 meters)
        depth_vis = np.clip(depth_image, 0, 2.0) / 2.0
        depth_vis = (depth_vis * 255).astype(np.uint8)
        cv2.imshow("Wrist Depth Camera", depth_vis)

        # Render and display RGB camera
        renderer_rgb.update_scene(data, camera="wrist_camera")
        rgb_image = renderer_rgb.render()
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Wrist RGB Camera", bgr_image)
        
        # Render and display Top camera
        renderer_top.update_scene(data, camera="top_camera")
        top_image = renderer_top.render()
        bgr_top = cv2.cvtColor(top_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Top RGB Camera", bgr_top)
        
        # Render and display Front camera
        renderer_front.update_scene(data, camera="front_camera")
        front_image = renderer_front.render()
        bgr_front = cv2.cvtColor(front_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Front RGB Camera", bgr_front)
        
        cv2.waitKey(1)

        # Maintain loop rate
        time.sleep(max(0, dt - 0.002))