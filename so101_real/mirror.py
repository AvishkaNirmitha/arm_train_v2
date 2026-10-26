"""Check that sim and real joint angles mean the same thing, and line up the real top camera.

The sim arm copies the real arm live. With --free the real arm's torque is off so you can move it by hand
(HOLD IT - it sags without torque). If a joint turns the other way in the sim, set its "sign" to -1 in
real_joint_map.json; if it is off by a constant angle, set "offset_deg". wrist_roll and the gripper aren't
mounted, so the sim keeps them at 0 / open.

With --top-cam N the real top camera is blended with the sim's top_camera view: move the real camera until
the mat, robot base and arm line up with the sim.

Run in .venv (arm_server.py must be running in .venv-lerobot):

    python so101_real/mirror.py --free
    python so101_real/mirror.py --free --top-cam      # camera 2
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from real_env import RealSetupEnv, GRIPPER_OPEN                          # noqa: E402
from real_arm_client import ArmClient, load_joint_map, real_to_sim       # noqa: E402


def open_camera(index, width=640, height=480):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} could not be opened")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", action="store_true", help="torque off so you can move the arm by hand")
    ap.add_argument("--top-cam", type=int, nargs="?", const=2, default=None,
                    help="overlay the real top camera on the sim view (--top-cam alone = camera 2)")
    args = ap.parse_args()

    joint_map = load_joint_map()
    arm = ArmClient()
    env = RealSetupEnv(randomize=False)
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    d.qpos[env.grip_qadr] = GRIPPER_OPEN
    renderer = mujoco.Renderer(m, height=480, width=640) if args.top_cam is not None else None
    cap = open_camera(args.top_cam) if args.top_cam is not None else None
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0

    if args.free:
        input("Torque goes OFF: hold the arm so it doesn't fall, then press ENTER ...")
        arm.free(True)
    print("Move the real arm; the sim copies it. Close the viewer (or Ctrl+C) to finish.")
    print("map:", {j: (v["sign"], v["offset_deg"]) for j, v in joint_map.items()})

    from mujoco import viewer
    try:
        with viewer.launch_passive(m, d) as v:
            v.cam.lookat[:] = [0.2, 0.0, 0.08]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.8, 200, -25
            while v.is_running():
                real = arm.read()["joints"]
                d.qpos[env.arm_qadr] = real_to_sim(real, joint_map)
                mujoco.mj_forward(m, d)
                v.sync()
                print("  ".join(f"{j} {real[j]:6.1f}" for j in real), end="\r")
                if cap is not None:
                    ok, frame = cap.read()
                    if ok:
                        renderer.update_scene(d, camera="top_camera", scene_option=opt)
                        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                        frame = cv2.resize(frame, (sim.shape[1], sim.shape[0]))
                        blend = cv2.addWeighted(frame, 0.5, sim, 0.5, 0)
                        try:
                            cv2.imshow("top camera: real (blended with) sim", blend)
                            cv2.waitKey(1)
                        except cv2.error:   # headless OpenCV: save the overlay instead
                            cv2.imwrite(os.path.join(HERE, "top_overlay.jpg"), blend)
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if args.free:
            input("Torque goes back ON at the current pose: hold the arm where it should stay, press ENTER ...")
            arm.free(False)
        arm.close()
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
