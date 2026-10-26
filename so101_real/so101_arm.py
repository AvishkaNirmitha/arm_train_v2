"""The real SO-101 follower arm, driven through lerobot with only the motors attached so far.

lerobot's `so101_follower` expects all 6 motors to answer when it connects. This wraps it with a motor list,
ATTACHED, so the arm runs without wrist_roll and the gripper until they are mounted. Its calibration file lives
under robots/so101_partial (not so101_follower), so the stock robot never picks up a partial calibration.

Run in .venv-lerobot (Python 3.12), from the project root:

    python so101_real/so101_arm.py setup                            # give ids to the ATTACHED motors, one at a time
    python so101_real/so101_arm.py setup --motors wrist_roll gripper  # later: only the new motors
    python so101_real/so101_arm.py calibrate
    python so101_real/so101_arm.py read                             # torque off, print joint angles (deg)

Adding wrist_roll and the gripper later: `setup --motors wrist_roll gripper` (each one alone on the board),
mount and chain them, add them to ATTACHED, then `calibrate` again (all motors are recalibrated together).
"""
import time
import argparse

from lerobot.motors import SerialMotorsBus
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

ALL_MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")  # ids 1..6
ATTACHED = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")  # add "wrist_roll", "gripper" when mounted

PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_arm"


class SO101Arm(SOFollower):
    name = "so101_partial"

    def __init__(self, config, motors=ATTACHED):
        super().__init__(config)
        # The bus builds its id tables on creation: make a new one with only these motors
        self.calibration = {m: c for m, c in self.calibration.items() if m in motors}
        self.bus = SerialMotorsBus(port=config.port, motors={m: self.bus.motors[m] for m in motors},
                                   calibration=self.calibration)

    def calibrate(self):
        # A file from before a motor was added doesn't cover it: recalibrate instead of offering the file
        if set(self.bus.motors) <= set(self.calibration) and self._keep_calibration_file(self.bus):
            return
        print("Video walkthrough: https://huggingface.co/docs/lerobot/main/en/so101#calibration-video")
        full_turn = [m for m in ("wrist_roll",) if m in self.bus.motors]
        self.calibration = self.bus.record_calibration("position", full_turn_motors=full_turn)
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)


def make_arm(port=PORT, robot_id=ROBOT_ID, motors=ATTACHED):
    return SO101Arm(SOFollowerRobotConfig(port=port, id=robot_id), motors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["setup", "calibrate", "read"])
    ap.add_argument("--motors", nargs="+", choices=ALL_MOTORS,
                    help="setup only: these motors (default ATTACHED), in this order")
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--id", default=ROBOT_ID)
    args = ap.parse_args()

    if args.command == "setup":
        # None: like lerobot-setup-motors, from the end of the arm down to the base
        make_arm(args.port, args.id, args.motors or ATTACHED).bus.setup_motors(args.motors)
        return

    arm = make_arm(args.port, args.id)
    arm.connect(calibrate=False)
    try:
        if args.command == "calibrate":
            arm.calibrate()
        else:
            if not arm.is_calibrated:
                raise SystemExit("Not calibrated: run `python so101_real/so101_arm.py calibrate` first.")
            arm.bus.disable_torque()   # move the arm by hand
            while True:
                obs = arm.get_observation()
                print("  ".join(f"{m}: {obs[f'{m}.pos']:7.1f}" for m in arm.bus.motors), end="\r")
                time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
